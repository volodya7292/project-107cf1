use crate::render_engine;
use crate::render_engine::material_pipeline::MaterialPipeline;
use crate::render_engine::scene::{ComponentStorage, Event};
use crate::render_engine::vertex_mesh::RawVertexMesh;
use crate::render_engine::{
    component, scene, BufferUpdate, BufferUpdate1, Renderable, Scene, VMBufferUpdate,
};
use crate::utils::HashMap;
use nalgebra as na;
use smallvec::{smallvec, SmallVec};
use std::collections::VecDeque;
use std::sync::{atomic, Arc, Mutex};
use std::time::Instant;
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{AccessFlags, WaitSemaphore};

pub(super) struct RendererCompEventsSystem<'a> {
    pub device: &'a Arc<vkw::Device>,
    pub renderer_comps: scene::LockedStorage<component::Renderer>,
    pub depth_per_object_pool: &'a mut vkw::DescriptorPool,
    pub g_per_pipeline_pools: &'a mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    pub renderables: &'a mut HashMap<u32, Renderable>,
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub material_pipelines: &'a [MaterialPipeline],
}

impl RendererCompEventsSystem<'_> {
    fn renderer_comp_created(
        renderable: &mut Renderable,
        depth_per_object_pool: &mut vkw::DescriptorPool,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        mat_pipes: &[MaterialPipeline],
    ) {
        renderable.descriptor_sets = smallvec![
            depth_per_object_pool.alloc().unwrap(),
            g_per_pipeline_pools
                .get_mut(&mat_pipes[renderable.material_pipe as usize].signature)
                .unwrap()
                .alloc()
                .unwrap(),
        ];
    }

    fn renderer_comp_modified(
        renderer: &mut component::Renderer,
        renderable: &mut Renderable,
        depth_per_object_pool: &mut vkw::DescriptorPool,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        buffer_updates: &mut Vec<BufferUpdate>,
        mat_pipes: &[MaterialPipeline],
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderable.descriptor_sets;

        depth_per_object_pool.update(
            inputs[0],
            &[vkw::Binding {
                id: 0,
                array_index: 0,
                res: vkw::BindingRes::Buffer(&renderable.buffers[0]),
            }],
        );

        let mut updates: SmallVec<[vkw::Binding; 4]> = smallvec![vkw::Binding {
            id: 0,
            array_index: 0,
            res: vkw::BindingRes::Buffer(&renderable.buffers[0]),
        }];

        for (binding_id, res) in &mut renderer.resources {
            if let component::renderer::Resource::Buffer(buf_res) = res {
                if buf_res.changed {
                    let data = mem::replace(&mut buf_res.buffer, vec![]);

                    buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                        buffer: buf_res.device_buffer.handle(),
                        offset: 0,
                        data,
                    }));
                    buf_res.changed = false;

                    updates.push(vkw::Binding {
                        id: *binding_id,
                        array_index: 0,
                        res: vkw::BindingRes::Buffer(&buf_res.device_buffer),
                    });
                }
            }
        }

        g_per_pipeline_pools
            .get_mut(mat_pipes[renderer.mat_pipeline as usize].signature())
            .unwrap()
            .update(inputs[1], &updates);
    }

    fn renderer_comp_removed(
        renderable: &Renderable,
        depth_per_object_pool: &mut vkw::DescriptorPool,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        mat_pipes: &[MaterialPipeline],
    ) {
        depth_per_object_pool.free(renderable.descriptor_sets[0]);
        g_per_pipeline_pools
            .get_mut(&mat_pipes[renderable.material_pipe as usize].signature)
            .unwrap()
            .free(renderable.descriptor_sets[1]);
    }

    pub fn run(&mut self) {
        let mut t0 = Instant::now();
        let mut renderer_comps = self.renderer_comps.write().unwrap();
        let events = renderer_comps.events();

        for event in events {
            match event {
                scene::Event::Created(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unchecked(entity).unwrap();
                    let pipe = &self.material_pipelines[renderer_comp.mat_pipeline as usize];
                    let uniform_buffer = self
                        .device
                        .create_device_buffer(
                            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::UNIFORM,
                            pipe.uniform_buffer_size() as u64,
                            1,
                        )
                        .unwrap();
                    let mut renderable = Renderable {
                        buffers: smallvec![uniform_buffer],
                        material_pipe: renderer_comp.mat_pipeline,
                        descriptor_sets: Default::default(),
                    };

                    Self::renderer_comp_created(
                        &mut renderable,
                        self.depth_per_object_pool,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut renderable,
                        self.depth_per_object_pool,
                        self.g_per_pipeline_pools,
                        self.buffer_updates,
                        self.material_pipelines,
                    );
                    self.renderables.insert(entity, renderable);
                }
                scene::Event::Modified(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unchecked(entity).unwrap();

                    let renderable = &self.renderables[&entity];
                    Self::renderer_comp_removed(
                        &renderable,
                        self.depth_per_object_pool,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );
                    self.renderables.remove(&entity);

                    let pipe = &self.material_pipelines[renderer_comp.mat_pipeline as usize];
                    let uniform_buffer = self
                        .device
                        .create_device_buffer(
                            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::UNIFORM,
                            pipe.uniform_buffer_size() as u64,
                            1,
                        )
                        .unwrap();
                    let mut renderable = Renderable {
                        buffers: smallvec![uniform_buffer],
                        material_pipe: renderer_comp.mat_pipeline,
                        descriptor_sets: Default::default(),
                    };
                    Self::renderer_comp_created(
                        &mut renderable,
                        self.depth_per_object_pool,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut renderable,
                        self.depth_per_object_pool,
                        self.g_per_pipeline_pools,
                        self.buffer_updates,
                        self.material_pipelines,
                    );
                    self.renderables.insert(entity, renderable);
                }
                Event::Removed(entity) => {
                    let renderable = &self.renderables[&entity];
                    Self::renderer_comp_removed(
                        &renderable,
                        self.depth_per_object_pool,
                        self.g_per_pipeline_pools,
                        self.material_pipelines,
                    );
                    self.renderables.remove(&entity);
                }
            }
        }

        let t1 = Instant::now();
        let t = (t1 - t0).as_secs_f64();
        if t > 0.003 {
            println!("renderer system {}", t);
        }
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
    pub buffer_updates: &'a mut Vec<BufferUpdate>,
    pub world_transform_comps: scene::LockedStorage<component::WorldTransform>,
    pub renderer_comps: scene::LockedStorage<component::Renderer>,
    pub renderables: &'a HashMap<u32, Renderable>,
}

impl WorldTransformEventsSystem<'_> {
    fn world_transform_modified(
        entity: u32,
        world_transform: &component::WorldTransform,
        renderer: Option<&component::Renderer>,
        buffer_updates: &mut Vec<BufferUpdate>,
        renderables: &HashMap<u32, Renderable>,
    ) {
        if let Some(renderer) = renderer {
            let matrix_bytes = unsafe {
                slice::from_raw_parts(
                    world_transform.matrix.as_ptr() as *const u8,
                    mem::size_of::<na::Matrix4<f32>>(),
                )
                .to_vec()
            };
            let renderable = &renderables[&entity];

            buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                buffer: renderable.buffers[0].handle(),
                offset: renderer.uniform_buffer_offset_model as u64,
                data: matrix_bytes,
            }));
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
                        self.buffer_updates,
                        self.renderables,
                    );
                }
                Event::Modified(entity) => {
                    Self::world_transform_modified(
                        entity,
                        world_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity),
                        self.buffer_updates,
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
}

impl CommitBufferUpdatesSystem<'_> {
    pub fn run(self) {
        for update in self.updates {
            self.vertex_meshes.insert(update.entity, update.mesh);
        }
    }
}
