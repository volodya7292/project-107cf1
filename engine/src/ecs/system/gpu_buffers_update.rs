use crate::ecs::component;
use crate::ecs::scene_storage;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity};
use crate::renderer::vertex_mesh::RawVertexMesh;
use crate::renderer::VMBufferUpdate;
use crate::utils::{HashMap, LruCache};
use parking_lot::Mutex;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::WaitSemaphore;

pub(crate) struct GpuBuffersUpdate<'a> {
    pub device: Arc<vkw::Device>,
    pub transfer_cl: &'a [Arc<Mutex<vkw::CmdList>>; 2],
    pub transfer_submit: &'a mut [vkw::SubmitPacket; 2],
    pub buffer_updates: &'a mut LruCache<Entity, Arc<RawVertexMesh>>,
    pub pending_buffer_updates: &'a mut Vec<VMBufferUpdate>,
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
            let mut transfer_barriers = Vec::with_capacity(self.buffer_updates.len());
            let mut graphics_barriers = Vec::with_capacity(transfer_barriers.len());

            for _ in 0..self.buffer_updates.len() {
                let (entity, mesh) = self.buffer_updates.pop_lru().unwrap();

                if mesh.staging_buffer.is_none() {
                    self.pending_buffer_updates.push(VMBufferUpdate { entity, mesh });
                    continue;
                }

                let src_buffer = mesh.staging_buffer.as_ref().unwrap();
                let dst_buffer = mesh.buffer.as_ref().unwrap();

                t_cl.copy_raw_host_buffer_to_device(&src_buffer.raw(), 0, dst_buffer, 0, src_buffer.size());

                transfer_barriers.push(
                    dst_buffer
                        .barrier()
                        .src_access_mask(vkw::AccessFlags::TRANSFER_WRITE)
                        .src_queue(transfer_queue)
                        .dst_queue(graphics_queue),
                );
                graphics_barriers.push(
                    dst_buffer
                        .barrier()
                        .src_queue(transfer_queue)
                        .dst_queue(graphics_queue),
                );

                total_copy_size += src_buffer.size();
                self.pending_buffer_updates.push(VMBufferUpdate { entity, mesh });

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
    pub vertex_meshes: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene_storage::LockedStorage<'a, component::VertexMesh>,
}

impl CommitBufferUpdates<'_> {
    pub fn run(self) {
        let vertex_mesh_comps = self.vertex_mesh_comps.read();

        for update in self.updates {
            if vertex_mesh_comps.contains(update.entity) {
                self.vertex_meshes.insert(update.entity, update.mesh);
            }
        }
    }
}
