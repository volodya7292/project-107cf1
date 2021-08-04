use crate::render_engine;
use crate::render_engine::material_pipeline::MaterialPipeline;
use crate::render_engine::scene::Event;
use crate::render_engine::vertex_mesh::RawVertexMesh;
use crate::render_engine::{
    component, scene, BufferUpdate, BufferUpdate1, Renderable, Scene, VMBufferUpdate,
};
use crate::utils::index_alloc::IndexAlloc;
use crate::utils::HashMap;
use nalgebra as na;
use smallvec::{smallvec, SmallVec};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{DescriptorSet, WaitSemaphore};

pub(super) struct RendererCompEventsSystem<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderer_comps: scene::LockedStorage<component::Renderer>,
    pub g_per_pipeline_pools: &'a mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    pub renderables: &'a mut HashMap<u32, Renderable>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a [MaterialPipeline],
    pub uniform_buffer_offsets: &'a mut IndexAlloc,
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
        mat_pipes: &[MaterialPipeline],
    ) {
        g_per_pipeline_pools
            .get_mut(&mat_pipes[renderable.material_pipe as usize].signature)
            .unwrap()
            .free(renderable.descriptor_sets[0]);
    }

    pub fn run(&mut self) {
        let mut renderer_comps = self.renderer_comps.write().unwrap();
        let events = renderer_comps.events();
        let mut binding_updates = Vec::<vkw::Binding>::with_capacity(events.len());
        let mut desc_updates = Vec::<(vkw::DescriptorSet, Range<usize>)>::with_capacity(events.len());

        for event in events {
            match event {
                scene::Event::Created(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unchecked(entity).unwrap();
                    let signature = &self.material_pipelines[renderer_comp.mat_pipeline as usize].signature;

                    let uniform_buf_index = self.uniform_buffer_offsets.alloc().unwrap();
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
                    let renderer_comp = renderer_comps.get_mut_unchecked(entity).unwrap();
                    let signature = &self.material_pipelines[renderer_comp.mat_pipeline as usize].signature;

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
                    self.uniform_buffer_offsets.free(renderable.uniform_buf_index);
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
    pub vertex_meshes: &'a mut HashMap<u32, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene::LockedStorage<component::VertexMesh>,
    pub buffer_updates: &'a mut VecDeque<VMBufferUpdate>,
}

impl VertexMeshCompEventsSystem<'_> {
    fn vertex_mesh_comp_modified(
        entity: u32,
        vertex_mesh_comp: &component::VertexMesh,
        buffer_updates: &mut VecDeque<VMBufferUpdate>,
    ) {
        let vertex_mesh = &vertex_mesh_comp.0;
        buffer_updates.push_back(VMBufferUpdate {
            entity,
            mesh: Arc::clone(vertex_mesh),
        });
    }

    pub fn run(&mut self) {
        let events = self.vertex_mesh_comps.write().unwrap().events();

        let vertex_mesh_comps = self.vertex_mesh_comps.read().unwrap();

        // Update device buffers of vertex meshes
        // ------------------------------------------------------------------------------------
        for event in &events {
            match event {
                scene::Event::Created(i) => {
                    Self::vertex_mesh_comp_modified(
                        *i,
                        vertex_mesh_comps.get(*i).unwrap(),
                        self.buffer_updates,
                    );
                }
                scene::Event::Modified(i) => {
                    // FIXME: the cause of spontaneous appearance of some mesh in different location:
                    // FIXME: Suppose an entity at index 0 was removed and immediately added, but with different transform.
                    // FIXME: => the rendering mesh stays the same until the new is transferred into GPU memory.
                    // FIXME: fix #1: upgrade entity indexing: add generational index to `Entity`
                    // FIXME: fix #2: ???

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

// Updates model transform matrices
pub(super) struct TransformEventsSystem {
    pub transform_comps: scene::LockedStorage<component::Transform>,
    pub model_transform_comps: scene::LockedStorage<component::ModelTransform>,
}

impl TransformEventsSystem {
    fn transform_modified(transform: &component::Transform, model_transform: &mut component::ModelTransform) {
        *model_transform = component::ModelTransform::from_transform(transform);
    }

    pub fn run(&mut self) {
        let events = self.transform_comps.write().unwrap().events();
        let transform_comps = self.transform_comps.read().unwrap();
        let mut model_transform_comps = self.model_transform_comps.write().unwrap();

        for event in events {
            match event {
                Event::Created(entity) => {
                    if !model_transform_comps.contains(entity) {
                        model_transform_comps.set(entity, component::ModelTransform::default());
                    }

                    Self::transform_modified(
                        transform_comps.get(entity).unwrap(),
                        model_transform_comps.get_mut(entity).unwrap(),
                    );
                }
                Event::Modified(entity) => {
                    if !model_transform_comps.contains(entity) {
                        model_transform_comps.set(entity, component::ModelTransform::default());
                    }

                    Self::transform_modified(
                        transform_comps.get(entity).unwrap(),
                        model_transform_comps.get_mut(entity).unwrap(),
                    );
                }
                _ => {}
            }
        }
    }
}

// Updates world transform uniform buffers
pub(super) struct WorldTransformEventsSystem<'a> {
    pub uniform_buffer_updates: &'a mut [BufferUpdate],
    pub world_transform_comps: scene::LockedStorage<component::WorldTransform>,
    pub renderer_comps: scene::LockedStorage<component::Renderer>,
    pub renderables: &'a HashMap<u32, Renderable>,
}

impl WorldTransformEventsSystem<'_> {
    fn world_transform_modified(
        entity: u32,
        world_transform: &component::WorldTransform,
        renderer: Option<&component::Renderer>,
        buffer_updates: &mut [BufferUpdate],
        renderables: &HashMap<u32, Renderable>,
    ) {
        if let Some(renderer) = renderer {
            let matrix_bytes = unsafe {
                slice::from_raw_parts(
                    world_transform.matrix.as_ptr() as *const u8,
                    mem::size_of::<na::Matrix4<f32>>(),
                )
            };
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
        let events = self.world_transform_comps.write().unwrap().events();
        let world_transform_comps = self.world_transform_comps.read().unwrap();
        let renderer_comps = self.renderer_comps.read().unwrap();

        for event in events {
            match event {
                Event::Created(entity) => {
                    Self::world_transform_modified(
                        entity,
                        world_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity),
                        self.uniform_buffer_updates,
                        self.renderables,
                    );
                }
                Event::Modified(entity) => {
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
pub(super) struct HierarchyPropagationSystem {
    pub parent_comps: scene::LockedStorage<component::Parent>,
    pub children_comps: scene::LockedStorage<component::Children>,
    pub model_transform_comps: scene::LockedStorage<component::ModelTransform>,
    pub world_transform_comps: scene::LockedStorage<component::WorldTransform>,
}

impl HierarchyPropagationSystem {
    fn propagate_hierarchy(
        parent_world_transform: component::WorldTransform,
        parent_world_transform_changed: bool,
        parent_entity: u32,
        entity: u32,
        parent_comps: &mut scene::ComponentStorageMut<component::Parent>,
        children_comps: &scene::ComponentStorage<component::Children>,
        model_transform_comps: &mut scene::ComponentStorageMut<component::ModelTransform>,
        world_transform_comps: &mut scene::ComponentStorageMut<component::WorldTransform>,
    ) {
        let model_transform = model_transform_comps.get_mut_unchecked(entity).unwrap();
        let world_transform_changed = parent_world_transform_changed || model_transform.changed;

        if model_transform.changed {
            model_transform.changed = false;
        }

        let world_transform = if world_transform_changed {
            let new_world_transform = parent_world_transform.combine(model_transform);
            world_transform_comps.set(entity, new_world_transform);
            new_world_transform
        } else {
            *world_transform_comps.get(entity).unwrap()
        };

        parent_comps.set(entity, component::Parent(parent_entity));

        if let Some(children) = children_comps.get(entity) {
            for &child in children.get() {
                Self::propagate_hierarchy(
                    world_transform,
                    world_transform_changed,
                    entity,
                    child,
                    parent_comps,
                    children_comps,
                    model_transform_comps,
                    world_transform_comps,
                );
            }
        }
    }

    pub fn run(&mut self) {
        let mut parent_comps = self.parent_comps.write().unwrap();
        let children_comps = self.children_comps.read().unwrap();
        let mut model_transform_comps = self.model_transform_comps.write().unwrap();
        let mut world_transform_comps = self.world_transform_comps.write().unwrap();

        // Collect global parents
        // !Parent & ModelTransform
        let entities: Vec<usize> = model_transform_comps
            .entries()
            .difference(parent_comps.entries())
            .collect();

        for entity in entities {
            let (model_transform_changed, world_transform) = {
                let model_transform = model_transform_comps.get_mut_unchecked(entity as u32).unwrap();
                let world_transform = component::WorldTransform::from_model_transform(&model_transform);
                let model_transform_changed = model_transform.changed;

                if model_transform_changed {
                    world_transform_comps.set(entity as u32, world_transform);
                    model_transform.changed = false;
                }

                (model_transform_changed, world_transform)
            };

            if let Some(children) = children_comps.get(entity as u32) {
                for &child in children.get() {
                    Self::propagate_hierarchy(
                        world_transform,
                        model_transform_changed,
                        entity as u32,
                        child,
                        &mut parent_comps,
                        &children_comps,
                        &mut model_transform_comps,
                        &mut world_transform_comps,
                    );
                }
            }
        }
    }
}

pub(super) struct BufferUpdateSystem<'a> {
    pub device: Arc<vkw::Device>,
    pub transfer_cl: &'a [Arc<Mutex<vkw::CmdList>>; 2],
    pub transfer_submit: &'a mut [vkw::SubmitPacket; 2],
    pub buffer_updates: &'a mut VecDeque<VMBufferUpdate>,
    pub pending_buffer_updates: &'a mut Vec<VMBufferUpdate>,
}

impl BufferUpdateSystem<'_> {
    const MAX_TRANSFER_SIZE_PER_RUN: u64 = 3145728; // 3M ~ 1ms

    pub fn run(&mut self) {
        let transfer_queue = self.device.get_queue(vkw::Queue::TYPE_TRANSFER);
        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);

        {
            let mut t_cl = self.transfer_cl[0].lock().unwrap();
            t_cl.begin(true).unwrap();
            let mut g_cl = self.transfer_cl[1].lock().unwrap();
            g_cl.begin(true).unwrap();

            let mut total_copy_size = 0;
            let mut transfer_barriers = Vec::with_capacity(self.buffer_updates.len());
            let mut graphics_barriers = Vec::with_capacity(transfer_barriers.len());

            for _ in 0..self.buffer_updates.len() {
                let update = self.buffer_updates.pop_front().unwrap();
                let mesh = &update.mesh;

                if mesh.staging_buffer.is_none() {
                    self.pending_buffer_updates.push(update);
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
                self.pending_buffer_updates.push(update);

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

        unsafe {
            transfer_queue.submit(&mut self.transfer_submit[0]).unwrap();

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
}

pub(super) struct CommitBufferUpdatesSystem<'a> {
    pub updates: Vec<VMBufferUpdate>,
    pub vertex_meshes: &'a mut HashMap<u32, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene::LockedStorage<component::VertexMesh>,
}

impl CommitBufferUpdatesSystem<'_> {
    pub fn run(self) {
        let vertex_mesh_comps = self.vertex_mesh_comps.read().unwrap();

        for update in self.updates {
            if vertex_mesh_comps.contains(update.entity) {
                self.vertex_meshes.insert(update.entity, update.mesh);
            }
        }
    }
}
