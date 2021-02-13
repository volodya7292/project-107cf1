use crate::renderer::scene::Event;
use crate::renderer::{component, scene, BufferUpdate, BufferUpdate1, BufferUpdate2, Renderable};
use crate::utils::HashMap;
use nalgebra as na;
use smallvec::{smallvec, SmallVec};
use std::sync::{atomic, Arc, Mutex};
use std::{mem, slice};
use vk_wrapper as vkw;

pub(super) struct RendererCompEventsSystem {
    pub renderer_comps: scene::LockedStorage<component::Renderer>,
    pub depth_per_object_pool: vkw::DescriptorPool,
    pub g_per_pipeline_pools: HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    pub renderables: Arc<Mutex<HashMap<u32, Renderable>>>,
    pub buffer_updates: Arc<Mutex<Vec<BufferUpdate>>>,
}

impl RendererCompEventsSystem {
    fn renderer_comp_created(
        renderer: &mut component::Renderer,
        depth_per_object_pool: &mut vkw::DescriptorPool,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    ) {
        renderer.descriptor_sets = vec![
            depth_per_object_pool.alloc().unwrap(),
            g_per_pipeline_pools
                .get_mut(renderer.mat_pipeline.signature())
                .unwrap()
                .alloc()
                .unwrap(),
        ];
    }

    fn renderer_comp_modified(
        renderer: &mut component::Renderer,
        depth_per_object_pool: &mut vkw::DescriptorPool,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
        buffer_updates: &mut Vec<BufferUpdate>,
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let inputs = &mut renderer.descriptor_sets;

        depth_per_object_pool.update(
            inputs[0],
            &[vkw::Binding {
                id: 0,
                array_index: 0,
                res: vkw::BindingRes::Buffer(Arc::clone(&renderer.uniform_buffer)),
            }],
        );

        let mut updates: SmallVec<[vkw::Binding; 4]> = smallvec![vkw::Binding {
            id: 0,
            array_index: 0,
            res: vkw::BindingRes::Buffer(Arc::clone(&renderer.uniform_buffer)),
        }];

        for (binding_id, res) in &mut renderer.resources {
            if let component::renderer::Resource::Buffer(buf_res) = res {
                if buf_res.changed {
                    let data = mem::replace(&mut buf_res.buffer, vec![]);

                    buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                        buffer: Arc::clone(&buf_res.device_buffer),
                        offset: 0,
                        data,
                    }));
                    buf_res.changed = false;

                    updates.push(vkw::Binding {
                        id: *binding_id,
                        array_index: 0,
                        res: vkw::BindingRes::Buffer(Arc::clone(&buf_res.device_buffer)),
                    });
                }
            }
        }

        g_per_pipeline_pools
            .get_mut(renderer.mat_pipeline.signature())
            .unwrap()
            .update(inputs[1], &updates);
    }

    fn renderer_comp_removed(
        renderer: &mut component::Renderer,
        depth_per_object_pool: &mut vkw::DescriptorPool,
        g_per_pipeline_pools: &mut HashMap<Arc<vkw::PipelineSignature>, vkw::DescriptorPool>,
    ) {
        depth_per_object_pool.free(renderer.descriptor_sets[0]);
        g_per_pipeline_pools
            .get_mut(renderer.mat_pipeline.signature())
            .unwrap()
            .free(renderer.descriptor_sets[2]);
    }

    pub fn run(&mut self) {
        let mut renderer_comps = self.renderer_comps.write().unwrap();
        let events = renderer_comps.events();
        let mut renderables = self.renderables.lock().unwrap();
        let mut buffer_updates = self.buffer_updates.lock().unwrap();

        for event in events {
            match event {
                scene::Event::Created(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unchecked(entity).unwrap();

                    Self::renderer_comp_created(
                        renderer_comp,
                        &mut self.depth_per_object_pool,
                        &mut self.g_per_pipeline_pools,
                    );
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut self.depth_per_object_pool,
                        &mut self.g_per_pipeline_pools,
                        &mut *buffer_updates,
                    );
                    renderables.insert(
                        entity,
                        Renderable {
                            buffers: smallvec![Arc::clone(&renderer_comp.uniform_buffer)],
                        },
                    );
                }
                scene::Event::Modified(entity) => {
                    let renderer_comp = renderer_comps.get_mut_unchecked(entity).unwrap();
                    Self::renderer_comp_modified(
                        renderer_comp,
                        &mut self.depth_per_object_pool,
                        &mut self.g_per_pipeline_pools,
                        &mut *buffer_updates,
                    );
                    renderables.get_mut(&entity).unwrap().buffers =
                        smallvec![Arc::clone(&renderer_comp.uniform_buffer)];
                }
                Event::Removed(entity, mut renderer) => {
                    Self::renderer_comp_removed(
                        &mut renderer,
                        &mut self.depth_per_object_pool,
                        &mut self.g_per_pipeline_pools,
                    );
                    renderables.remove(&entity);
                }
            }
        }
    }
}

pub(super) struct VertexMeshCompEventsSystem {
    pub vertex_mesh_comps: scene::LockedStorage<component::VertexMesh>,
    pub buffer_updates: Arc<Mutex<Vec<BufferUpdate>>>,
}

impl VertexMeshCompEventsSystem {
    fn vertex_mesh_comp_modified(
        vertex_mesh_comp: &component::VertexMesh,
        buffer_updates: &mut Vec<BufferUpdate>,
    ) {
        let vertex_mesh = &vertex_mesh_comp.0;

        if vertex_mesh.changed.swap(false, atomic::Ordering::Relaxed) {
            let staging_buffer = vertex_mesh.staging_buffer.as_ref().unwrap();

            buffer_updates.push(BufferUpdate::Type2(BufferUpdate2 {
                src_buffer: staging_buffer.raw(),
                src_offset: 0,
                dst_buffer: Arc::clone(vertex_mesh.buffer.as_ref().unwrap()),
                dst_offset: 0,
                size: staging_buffer.size(),
            }));
        }
    }

    pub fn run(&mut self) {
        let events = self.vertex_mesh_comps.write().unwrap().events();

        let vertex_mesh_comps = self.vertex_mesh_comps.read().unwrap();
        let mut buffer_updates = self.buffer_updates.lock().unwrap();

        // Update device buffers of vertex meshes
        // ------------------------------------------------------------------------------------
        for event in &events {
            match event {
                scene::Event::Created(i) => {
                    Self::vertex_mesh_comp_modified(vertex_mesh_comps.get(*i).unwrap(), &mut *buffer_updates);
                }
                scene::Event::Modified(i) => {
                    Self::vertex_mesh_comp_modified(vertex_mesh_comps.get(*i).unwrap(), &mut *buffer_updates);
                }
                _ => {}
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
pub(super) struct WorldTransformEventsSystem {
    pub buffer_updates: Arc<Mutex<Vec<BufferUpdate>>>,
    pub world_transform_comps: scene::LockedStorage<component::WorldTransform>,
    pub renderer_comps: scene::LockedStorage<component::Renderer>,
}

impl WorldTransformEventsSystem {
    fn world_transform_modified(
        world_transform: &component::WorldTransform,
        renderer: Option<&component::Renderer>,
        buffer_updates: &mut Vec<BufferUpdate>,
    ) {
        if let Some(renderer) = renderer {
            let matrix_bytes = unsafe {
                slice::from_raw_parts(
                    world_transform.matrix.as_ptr() as *const u8,
                    mem::size_of::<na::Matrix4<f32>>(),
                )
                .to_vec()
            };

            buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                buffer: Arc::clone(&renderer.uniform_buffer),
                offset: renderer.mat_pipeline.uniform_buffer_offset_model() as u64,
                data: matrix_bytes,
            }));
        }
    }

    pub fn run(&mut self) {
        let events = self.world_transform_comps.write().unwrap().events();
        let world_transform_comps = self.world_transform_comps.read().unwrap();
        let renderer_comps = self.renderer_comps.read().unwrap();
        let mut buffer_updates = self.buffer_updates.lock().unwrap();

        for event in events {
            match event {
                Event::Created(entity) => {
                    Self::world_transform_modified(
                        world_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity),
                        &mut buffer_updates,
                    );
                }
                Event::Modified(entity) => {
                    Self::world_transform_modified(
                        world_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity),
                        &mut buffer_updates,
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
            assert_eq!(children.0.len(), 0);
            for child in &children.0 {
                Self::propagate_hierarchy(
                    world_transform,
                    world_transform_changed,
                    entity,
                    *child,
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
                for &child in &children.0 {
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
