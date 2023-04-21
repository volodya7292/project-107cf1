use crate::ecs::component::VertexMeshC;
use crate::module::main_renderer::gpu_executor::{GPUJobDeviceExt, GPUJobExecInfo};
use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use crate::module::main_renderer::ParallelJob;
use common::types::{HashMap, HashSet};
use entity_data::{EntityId, SystemAccess, SystemHandler};
use std::sync::Arc;
use std::time::Instant;
use vk_wrapper as vkw;

const MAX_TRANSFER_SIZE_PER_RUN: u64 = 3145728; // 3M ~ 1ms

pub(crate) struct GpuBuffersUpdate<'a> {
    pub device: Arc<vkw::Device>,
    pub transfer_jobs: &'a mut ParallelJob,
    pub buffer_updates: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub sorted_buffer_updates_entities: &'a Vec<(EntityId, f32)>,
    pub pending_buffer_updates: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub run_time: f64,
}

impl SystemHandler for GpuBuffersUpdate<'_> {
    fn run(&mut self, _: SystemAccess) {
        let t0 = Instant::now();
        let transfer_queue = self.device.get_queue(vkw::QueueType::Transfer);
        let graphics_queue = self.device.get_queue(vkw::QueueType::Graphics);

        self.transfer_jobs.work.wait().unwrap();
        self.transfer_jobs.sync.wait().unwrap();

        let mut processed_meshes = HashSet::new();

        {
            let t_cl = self.transfer_jobs.work.get_cmd_list_for_recording();
            let g_cl = self.transfer_jobs.sync.get_cmd_list_for_recording();

            t_cl.begin(true).unwrap();
            g_cl.begin(true).unwrap();

            let mut total_copy_size = 0;
            let mut transfer_barriers = Vec::with_capacity(self.buffer_updates.len());
            let mut graphics_barriers = Vec::with_capacity(transfer_barriers.len());

            for (entity, _) in self.sorted_buffer_updates_entities {
                let mesh = self.buffer_updates.remove(entity).unwrap();

                if mesh.staging_buffer.is_none() {
                    self.pending_buffer_updates.insert(*entity, mesh);
                    continue;
                }

                let src_buffer = mesh.staging_buffer.as_ref().unwrap();
                let dst_buffer = mesh.buffer.as_ref().unwrap();

                // Multiple entities may use the same mesh, process it only once
                if !processed_meshes.insert(src_buffer.as_ptr()) {
                    continue;
                }

                t_cl.copy_buffer(src_buffer, 0, dst_buffer, 0, src_buffer.size());

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
                self.pending_buffer_updates.insert(*entity, mesh);

                if total_copy_size >= MAX_TRANSFER_SIZE_PER_RUN {
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

        unsafe {
            self.device
                .run_jobs(&mut [GPUJobExecInfo::new(&mut self.transfer_jobs.work)])
                .unwrap()
        };

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}

pub(crate) struct CommitBufferUpdates<'a> {
    pub updates: HashMap<EntityId, Arc<RawVertexMesh>>,
    pub vertex_meshes: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub run_time: f64,
}

impl SystemHandler for CommitBufferUpdates<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let vertex_mesh_comps = data.component::<VertexMeshC>();

        for (entity, updated_mesh) in self.updates.drain() {
            if vertex_mesh_comps.contains(&entity) {
                self.vertex_meshes.insert(entity, updated_mesh);
            }
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
