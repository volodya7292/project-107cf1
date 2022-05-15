use crate::ecs::component;
use crate::ecs::scene::Entity;
use crate::ecs::scene_storage::{ComponentStorageImpl, LockedStorage};
use crate::renderer::vertex_mesh::RawVertexMesh;
use crate::renderer::{GBVertexMesh, VMBufferUpdate};
use crate::utils::HashMap;
use parking_lot::Mutex;
use range_alloc::RangeAllocator;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::{DeviceBuffer, WaitSemaphore};

pub(crate) struct GpuBuffersUpdate<'a> {
    pub device: Arc<vkw::Device>,
    pub transfer_cl: &'a [Arc<Mutex<vkw::CmdList>>; 2],
    pub transfer_submit: &'a mut [vkw::SubmitPacket; 2],
    pub vertex_mesh_updates: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub pending_buffer_updates: &'a mut Vec<VMBufferUpdate>,
    pub global_buffer: &'a DeviceBuffer,
    pub gb_vertex_meshes: &'a mut HashMap<usize, GBVertexMesh>,
    pub gb_allocator: &'a mut RangeAllocator<u64>,
}

impl GpuBuffersUpdate<'_> {
    const MAX_TRANSFER_SIZE_PER_RUN: u64 = 3145728; // 3M ~ 1ms

    pub fn run(&mut self) {
        let transfer_queue = self.device.get_queue(vkw::Queue::TYPE_TRANSFER);
        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);

        self.transfer_submit[0].wait().unwrap();
        self.transfer_submit[1].wait().unwrap();

        {
            let mut t_cl = self.transfer_cl[0].lock();
            let mut g_cl = self.transfer_cl[1].lock();

            t_cl.begin(true).unwrap();
            g_cl.begin(true).unwrap();

            let mut total_copy_size = 0;
            let mut transfer_barriers = Vec::with_capacity(self.vertex_mesh_updates.len());
            let mut graphics_barriers = Vec::with_capacity(transfer_barriers.len());

            for _ in 0..self.vertex_mesh_updates.len() {
                let entity = *self.vertex_mesh_updates.keys().next().unwrap();
                let raw_mesh = self.vertex_mesh_updates.remove(&entity).unwrap();
                let src_buffer = raw_mesh.staging_buffer.as_ref().unwrap();
                let mesh_ptr = src_buffer.as_ptr() as usize;

                if self.gb_vertex_meshes.contains_key(&mesh_ptr) {
                    self.pending_buffer_updates
                        .push(VMBufferUpdate { entity, mesh_ptr });
                    // The vertex mesh update is already processed (multiple entities may use the same mesh)
                    continue;
                }

                let gb_range = self.gb_allocator.allocate_range(src_buffer.size()).unwrap();
                let gb_mesh = GBVertexMesh {
                    raw: Arc::clone(&raw_mesh),
                    ref_count: 1,
                    gb_range: gb_range.clone(),
                    gb_binding_offsets: raw_mesh
                        .binding_offsets
                        .iter()
                        .map(|v| gb_range.start + *v)
                        .collect(),
                    gb_indices_offset: gb_range.start + raw_mesh.indices_offset,
                };
                let gb_range = &gb_mesh.gb_range;

                t_cl.copy_raw_host_buffer_to_device(
                    &src_buffer.raw(),
                    0,
                    self.global_buffer,
                    gb_range.start,
                    src_buffer.size(),
                );

                transfer_barriers.push(
                    self.global_buffer
                        .barrier()
                        .src_access_mask(vkw::AccessFlags::TRANSFER_WRITE)
                        .offset(gb_range.start)
                        .size(gb_range.end - gb_range.start)
                        .src_queue(transfer_queue)
                        .dst_queue(graphics_queue),
                );
                graphics_barriers.push(
                    self.global_buffer
                        .barrier()
                        .offset(gb_range.start)
                        .size(gb_range.end - gb_range.start)
                        .src_queue(transfer_queue)
                        .dst_queue(graphics_queue),
                );

                self.gb_vertex_meshes.insert(mesh_ptr, gb_mesh);
                self.pending_buffer_updates
                    .push(VMBufferUpdate { entity, mesh_ptr });

                total_copy_size += src_buffer.size();

                if total_copy_size >= Self::MAX_TRANSFER_SIZE_PER_RUN {
                    break;
                }
            }

            t_cl.barrier_buffer(
                vkw::PipelineStageFlags::TRANSFER,
                vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                &transfer_barriers,
            );
            g_cl.barrier_buffer(
                vkw::PipelineStageFlags::TOP_OF_PIPE,
                vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                &graphics_barriers,
            );

            t_cl.end().unwrap();
            g_cl.end().unwrap();
        }

        unsafe { transfer_queue.submit(&mut self.transfer_submit[0]).unwrap() };

        self.transfer_submit[1]
            .set(&[vkw::SubmitInfo::new(
                &[WaitSemaphore {
                    semaphore: Arc::clone(transfer_queue.timeline_semaphore()),
                    wait_dst_mask: vkw::PipelineStageFlags::TRANSFER,
                    wait_value: self.transfer_submit[0].get_signal_value(0).unwrap(),
                }],
                &[Arc::clone(&self.transfer_cl[1])],
                &[],
            )])
            .unwrap();
    }
}

pub(crate) struct CommitBufferUpdates<'a> {
    pub updates: Vec<VMBufferUpdate>,
    pub entity_vertex_meshes: &'a mut HashMap<Entity, usize>,
    pub to_remove_vertex_meshes: &'a mut HashMap<usize, u32>,
    pub vertex_mesh_comps: LockedStorage<'a, component::VertexMesh>,
}

impl CommitBufferUpdates<'_> {
    pub fn run(self) {
        let vertex_mesh_comps = self.vertex_mesh_comps.read();

        for update in self.updates {
            // Check if update is still relevant (some entity is still using it)
            if vertex_mesh_comps.contains(update.entity) {
                let prev_mesh_ptr = self.entity_vertex_meshes.insert(update.entity, update.mesh_ptr);

                if let Some(prev_mesh_ptr) = prev_mesh_ptr {
                    let remove_refs_n = self.to_remove_vertex_meshes.entry(prev_mesh_ptr).or_insert(0);
                    *remove_refs_n += 1;
                }
            }
        }
    }
}
