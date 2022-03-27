use crate::render_engine;
use crate::render_engine::material_pipeline::MaterialPipelineSet;
use crate::render_engine::scene::{ComponentStorageImpl, Entity, Event};
use crate::render_engine::vertex_mesh::RawVertexMesh;
use crate::render_engine::{component, scene, BufferUpdate, BufferUpdate1, Renderable, VMBufferUpdate};
use crate::utils::{HashMap, LruCache};
use index_pool::IndexPool;
use nalgebra_glm::Mat4;
use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};
use std::ops::Range;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{DescriptorSet, WaitSemaphore};

pub(super) struct RendererCompEventsSystem<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderer_comps: scene::LockedStorage<'a, component::Renderer>,
    pub g_per_pipeline_pools: &'a mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    pub renderables: &'a mut HashMap<Entity, Renderable>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a [MaterialPipelineSet],
    pub uniform_buffer_offsets: &'a mut IndexPool,
}

impl RendererCompEventsSystem<'_> {
    fn renderer_comp_created(
        renderable: &mut Renderable,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        signature: &vkw::PipelineSignature,
    ) {
        renderable.descriptor_sets =
            smallvec![g_per_pipeline_pools.get_mut(signature).unwrap().alloc().unwrap(),];
    }

    fn renderer_comp_modified(
        renderer: &mut component::Renderer,
        renderable: &mut Renderable,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        buffer_updates: &mut Vec<BufferUpdate>,
        signature: &vkw::PipelineSignature,
        binding_updates: &mut Vec<vkw::Binding>,
        desc_updates: &mut Vec<(DescriptorSet, Range<usize>)>,
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderable.descriptor_sets;

        let g_pool = g_per_pipeline_pools.get_mut(signature).unwrap();
        let mut updates: SmallVec<[vkw::Binding; 4]> = smallvec![];

        for (binding_id, res) in &mut renderer.resources {
            if let component::renderer::Resource::Buffer(buf_res) = res {
                if buf_res.changed {
                    let data = mem::take(&mut buf_res.buffer);

                    buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                        buffer: buf_res.device_buffer.handle(),
                        offset: 0,
                        data,
                    }));
                    buf_res.changed = false;

                    updates.push(g_pool.create_binding(
                        *binding_id,
                        0,
                        vkw::BindingRes::Buffer(buf_res.device_buffer.handle()),
                    ));
                }
            }
        }

        let s0 = binding_updates.len();
        binding_updates.extend(updates);
        let s = s0..binding_updates.len();
        if !s.is_empty() {
            desc_updates.push((inputs[0], s));
        }
    }

    fn renderer_comp_removed(
        renderable: &Renderable,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        mat_pipes: &[MaterialPipelineSet],
    ) {
        g_per_pipeline_pools
            .get_mut(&mat_pipes[renderable.material_pipe as usize].main_signature)
            .unwrap()
            .free(renderable.descriptor_sets[0]);
    }

    pub fn run(&mut self) {
        let mut renderer_comps = self.renderer_comps.write();
        let events = renderer_comps.events();
        let mut binding_updates = Vec::<vkw::Binding>::with_capacity(events.len());
        let mut desc_updates = Vec::<(vkw::DescriptorSet, Range<usize>)>::with_capacity(events.len());

        for event in events {
            match event {
                scene::Event::Created(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unmarked(entity).unwrap();
                    let signature =
                        &self.material_pipelines[renderer_comp.mat_pipeline as usize].main_signature;

                    let uniform_buf_index = self.uniform_buffer_offsets.new_id();
                    let mut renderable = Renderable {
                        buffers: smallvec![],
                        material_pipe: renderer_comp.mat_pipeline,
                        uniform_buf_index,
                        descriptor_sets: Default::default(),
                    };

                    Self::renderer_comp_created(&mut renderable, self.g_per_pipeline_pools, signature);
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut renderable,
                        self.g_per_pipeline_pools,
                        self.buffer_updates,
                        signature,
                        &mut binding_updates,
                        &mut desc_updates,
                    );
                    self.renderables.insert(entity, renderable);
                }
                scene::Event::Modified(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unmarked(entity).unwrap();
                    let signature =
                        &self.material_pipelines[renderer_comp.mat_pipeline as usize].main_signature;

                    let mut renderable = self.renderables.remove(&entity).unwrap();
                    Self::renderer_comp_removed(
                        &renderable,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );

                    renderable.buffers = smallvec![];
                    renderable.material_pipe = renderer_comp.mat_pipeline;
                    renderable.descriptor_sets = Default::default();

                    Self::renderer_comp_created(&mut renderable, self.g_per_pipeline_pools, signature);
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut renderable,
                        self.g_per_pipeline_pools,
                        self.buffer_updates,
                        signature,
                        &mut binding_updates,
                        &mut desc_updates,
                    );
                    self.renderables.insert(entity, renderable);
                }
                Event::Removed(entity) => {
                    let renderable = &self.renderables[&entity];
                    Self::renderer_comp_removed(
                        &renderable,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );
                    self.uniform_buffer_offsets
                        .return_id(renderable.uniform_buf_index)
                        .unwrap();
                    self.renderables.remove(&entity);
                }
            }
        }

        unsafe {
            self.device
                .update_descriptor_sets(&binding_updates, &desc_updates)
        };
    }
}

pub(super) struct VertexMeshCompEventsSystem<'a> {
    pub vertex_meshes: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene::LockedStorage<'a, component::VertexMesh>,
    pub buffer_updates: &'a mut LruCache<Entity, Arc<RawVertexMesh>>,
}

impl VertexMeshCompEventsSystem<'_> {
    fn vertex_mesh_comp_modified(
        entity: Entity,
        vertex_mesh_comp: &component::VertexMesh,
        buffer_updates: &mut LruCache<Entity, Arc<RawVertexMesh>>,
    ) {
        let vertex_mesh = &vertex_mesh_comp.0;
        buffer_updates.put(entity, Arc::clone(vertex_mesh));
    }

    pub fn run(&mut self) {
        let events = self.vertex_mesh_comps.write().events();

        let vertex_mesh_comps = self.vertex_mesh_comps.read();

        // Update device buffers of vertex meshes
        // ------------------------------------------------------------------------------------
        for event in &events {
            match event {
                scene::Event::Created(i) | scene::Event::Modified(i) => {
                    Self::vertex_mesh_comp_modified(
                        *i,
                        vertex_mesh_comps.get(*i).unwrap(),
                        self.buffer_updates,
                    );
                }
                Event::Removed(i) => {
                    self.vertex_meshes.remove(i);
                }
            }
        }
    }
}

// Updates world transform uniform buffers
pub(super) struct WorldTransformEventsSystem<'a> {
    pub uniform_buffer_updates: &'a mut [BufferUpdate],
    pub world_transform_comps: scene::LockedStorage<'a, component::WorldTransform>,
    pub renderer_comps: scene::LockedStorage<'a, component::Renderer>,
    pub renderables: &'a HashMap<Entity, Renderable>,
}

impl WorldTransformEventsSystem<'_> {
    fn world_transform_modified(
        entity: Entity,
        world_transform: &component::WorldTransform,
        renderer: Option<&component::Renderer>,
        buffer_updates: &mut [BufferUpdate],
        renderables: &HashMap<Entity, Renderable>,
    ) {
        if let Some(renderer) = renderer {
            let matrix = world_transform.matrix_f32();
            let matrix_bytes =
                unsafe { slice::from_raw_parts(matrix.as_ptr() as *const u8, mem::size_of::<Mat4>()) };
            let renderable = &renderables[&entity];

            if let BufferUpdate::Type2(upd) = &mut buffer_updates[0] {
                let src_offset = upd.data.len();
                upd.data.extend(matrix_bytes);
                upd.regions.push(vkw::CopyRegion::new(
                    src_offset as u64,
                    renderable.uniform_buf_index as u64 * render_engine::MAX_BASIC_UNIFORM_BLOCK_SIZE
                        + renderer.uniform_buffer_offset_model as u64,
                    matrix_bytes.len() as u64,
                ));
            } else {
                unreachable!()
            }
        }
    }

    pub fn run(&mut self) {
        let events = self.world_transform_comps.write().events();
        let world_transform_comps = self.world_transform_comps.read();
        let renderer_comps = self.renderer_comps.read();

        for event in events {
            match event {
                Event::Created(entity) | Event::Modified(entity) => {
                    Self::world_transform_modified(
                        entity,
                        world_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity),
                        self.uniform_buffer_updates,
                        self.renderables,
                    );
                }
                _ => {}
            }
        }
    }
}

// Propagates transform hierarchy and calculates world transforms
pub(super) struct HierarchyPropagationSystem<'a> {
    pub parent_comps: scene::LockedStorage<'a, component::Parent>,
    pub children_comps: scene::LockedStorage<'a, component::Children>,
    pub transform_comps: scene::LockedStorage<'a, component::Transform>,
    pub world_transform_comps: scene::LockedStorage<'a, component::WorldTransform>,
    pub ordered_entities: &'a mut Vec<Entity>,
}

struct StackEntry {
    entity: Entity,
    parent_world_transform_changed: bool,
    parent_world_transform: component::WorldTransform,
}

impl HierarchyPropagationSystem<'_> {
    pub fn run(&mut self) {
        let parent_comps = self.parent_comps.read();
        let children_comps = self.children_comps.read();
        let mut transform_comps = self.transform_comps.write();
        let mut world_transform_comps = self.world_transform_comps.write();
        let mut stack = Vec::<StackEntry>::with_capacity(transform_comps.len());

        // Collect global parents
        // !Parent & Transform (global parent entity doesn't have a Parent component)
        let entities = transform_comps.entries().difference(&parent_comps);
        stack.extend(entities.iter().map(|e| StackEntry {
            entity: e,
            parent_world_transform_changed: false,
            parent_world_transform: Default::default(),
        }));

        self.ordered_entities.clear();

        // Recursion using loop
        while let Some(StackEntry {
            entity,
            parent_world_transform_changed: parent_transform_changed,
            parent_world_transform,
        }) = stack.pop()
        {
            // Maybe this entity is dead (was removed but not removed from parent's `Children` component)
            if !transform_comps.contains(entity) {
                continue;
            }

            self.ordered_entities.push(entity);

            let world_transform_changed = parent_transform_changed || transform_comps.was_modified(entity);

            let world_transform = if world_transform_changed {
                let model_transform = transform_comps.get(entity).unwrap();

                let new_world_transform: component::WorldTransform =
                    parent_world_transform.combine(model_transform).into();

                world_transform_comps.set(entity, new_world_transform);
                new_world_transform
            } else {
                *world_transform_comps.get(entity).unwrap()
            };

            if let Some(children) = children_comps.get(entity) {
                // Because we're popping from the stack, insert in reversed order
                // to preserve the right order of insertion to `ordered_entities`
                stack.extend(children.children.iter().rev().map(|e| StackEntry {
                    entity: *e,
                    parent_world_transform_changed: world_transform_changed,
                    parent_world_transform: world_transform,
                }));
            }
        }

        transform_comps.clear_events();
    }
}

pub(super) struct BufferUpdateSystem<'a> {
    pub device: Arc<vkw::Device>,
    pub transfer_cl: &'a [Arc<Mutex<vkw::CmdList>>; 2],
    pub transfer_submit: &'a mut [vkw::SubmitPacket; 2],
    pub buffer_updates: &'a mut LruCache<Entity, Arc<RawVertexMesh>>,
    pub pending_buffer_updates: &'a mut Vec<VMBufferUpdate>,
}

impl BufferUpdateSystem<'_> {
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

pub(super) struct CommitBufferUpdatesSystem<'a> {
    pub updates: Vec<VMBufferUpdate>,
    pub vertex_meshes: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene::LockedStorage<'a, component::VertexMesh>,
}

impl CommitBufferUpdatesSystem<'_> {
    pub fn run(self) {
        let vertex_mesh_comps = self.vertex_mesh_comps.read();

        for update in self.updates {
            if vertex_mesh_comps.contains(update.entity) {
                self.vertex_meshes.insert(update.entity, update.mesh);
            }
        }
    }
}
