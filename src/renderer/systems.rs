use crate::renderer::scene::Event;
use crate::renderer::{component, scene, BufferUpdate, DistanceSortedRenderables};
use nalgebra as na;
use std::sync::{atomic, Arc, Mutex, RwLock};
use std::{cmp, mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::SubmitPacket;

pub(super) struct RendererCompEventsSystem {
    pub renderer_comps: Arc<RwLock<scene::ComponentStorage<component::Renderer>>>,
    pub sorted_renderables: Arc<Mutex<DistanceSortedRenderables>>,
    pub depth_per_object_pool: Arc<vkw::DescriptorPool>,
    pub model_inputs: Arc<vkw::DescriptorPool>,
}

impl RendererCompEventsSystem {
    fn renderer_comp_modified(
        renderer: &mut component::Renderer,
        depth_per_object_pool: &Arc<vkw::DescriptorPool>,
        model_inputs: &Arc<vkw::DescriptorPool>,
    ) {
        // Update pipeline inputs
        // ------------------------------------------------------------------------------------------
        let _mat_pipeline = &renderer.mat_pipeline;
        let inputs = &mut renderer.pipeline_inputs;

        inputs.clear();

        let depth_per_object = depth_per_object_pool.allocate_input().unwrap();
        depth_per_object.update(&[vkw::Binding {
            id: 0,
            array_index: 0,
            res: vkw::BindingRes::Buffer(Arc::clone(&renderer.uniform_buffer)),
        }]);

        let uniform_input = model_inputs.allocate_input().unwrap();
        uniform_input.update(&[vkw::Binding {
            id: 0,
            array_index: 0,
            res: vkw::BindingRes::Buffer(Arc::clone(&renderer.uniform_buffer)),
        }]);

        inputs.extend_from_slice(&[depth_per_object, uniform_input]);
    }

    pub fn run(&mut self) {
        let mut renderer_comps = self.renderer_comps.write().unwrap();
        let mut dsr = self.sorted_renderables.lock().unwrap();
        let sorted_renderables = &mut dsr.entities;
        let mut removed_count = 1;

        // Add new objects to sort
        // -------------------------------------------------------------------------------------------------------------
        {
            let events = renderer_comps.events();

            for event in events {
                match event {
                    scene::Event::Created(i) => {
                        Self::renderer_comp_modified(
                            renderer_comps.get_mut_unchecked(i).unwrap(),
                            &self.depth_per_object_pool,
                            &self.model_inputs,
                        );
                        sorted_renderables.push(i);
                    }
                    scene::Event::Modified(i) => {
                        Self::renderer_comp_modified(
                            renderer_comps.get_mut_unchecked(i).unwrap(),
                            &self.depth_per_object_pool,
                            &self.model_inputs,
                        );
                    }
                    scene::Event::Removed(_) => {
                        removed_count += 1;
                    }
                }
            }
        }

        // Replace removed(dead) entities with alive ones
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut swap_entities = Vec::<u32>::with_capacity(removed_count);
            let mut new_len = sorted_renderables.len();

            // Find alive entities for replacement
            for &entity in sorted_renderables.iter().rev() {
                if renderer_comps.is_alive(entity) {
                    if removed_count > swap_entities.len() {
                        swap_entities.push(entity);
                    } else {
                        break;
                    }
                }
                new_len -= 1;
            }

            // Resize vector to trim swapped entities
            sorted_renderables.truncate(new_len);

            // Swap entities
            for entity in sorted_renderables.iter_mut() {
                if !renderer_comps.is_alive(*entity) {
                    *entity = swap_entities.remove(swap_entities.len() - 1);
                }
            }

            // Add the rest of swap_entities that were not swapped due to resized vector
            sorted_renderables.extend(swap_entities);
        }
    }
}

pub(super) struct VertexMeshCompEventsSystem {
    pub events: Vec<scene::Event>,
    pub vertex_mesh_comps: Arc<RwLock<scene::ComponentStorage<component::VertexMesh>>>,
    pub device: Arc<vkw::Device>,
    pub staging_cl: Arc<Mutex<vkw::CmdList>>,
    pub staging_submit: Arc<Mutex<SubmitPacket>>,
}

impl VertexMeshCompEventsSystem {
    fn vertex_mesh_comp_modified(vertex_mesh_comp: &component::VertexMesh, cl: &mut vkw::CmdList) {
        let vertex_mesh = &vertex_mesh_comp.0;

        if vertex_mesh.changed.load(atomic::Ordering::Relaxed) {
            cl.copy_buffer_to_device(
                vertex_mesh.staging_buffer.as_ref().unwrap(),
                0,
                vertex_mesh.buffer.as_ref().unwrap(),
                0,
                vertex_mesh.staging_buffer.as_ref().unwrap().size(),
            );

            vertex_mesh.changed.store(false, atomic::Ordering::Relaxed);
        }
    }

    pub fn run(&mut self) {
        let vertex_mesh_comps = self.vertex_mesh_comps.read().unwrap();
        let mut submit = self.staging_submit.lock().unwrap();
        submit.wait().unwrap();

        // Update device buffers of vertex meshes
        // ------------------------------------------------------------------------------------
        {
            let mut cl = self.staging_cl.lock().unwrap();
            cl.begin(true).unwrap();

            for &event in &self.events {
                match event {
                    scene::Event::Created(i) => {
                        Self::vertex_mesh_comp_modified(vertex_mesh_comps.get(i).unwrap(), &mut *cl);
                    }
                    scene::Event::Modified(i) => {
                        Self::vertex_mesh_comp_modified(vertex_mesh_comps.get(i).unwrap(), &mut *cl);
                    }
                    scene::Event::Removed(_) => {}
                }
            }

            cl.end().unwrap();
        }

        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);
        graphics_queue.submit(&mut submit).unwrap();
    }
}

// Sort render objects from front to back (for Z rejection & occlusion queries)
pub(super) struct DistanceSortSystem {
    pub world_transform_comps: Arc<RwLock<scene::ComponentStorage<component::WorldTransform>>>,
    pub vertex_mesh_comps: Arc<RwLock<scene::ComponentStorage<component::VertexMesh>>>,
    pub sorted_renderables: Arc<Mutex<DistanceSortedRenderables>>,
    pub camera_pos: na::Vector3<f32>,
}

impl DistanceSortSystem {
    const DISTANCE_SORT_PER_UPDATE: usize = 128;

    pub fn run(&mut self) {
        let transform_comps = self.world_transform_comps.read().unwrap();
        let vertex_mesh_comps = self.vertex_mesh_comps.read().unwrap();
        let mut dsr = self.sorted_renderables.lock().unwrap();

        let curr_sort_count = dsr.curr_sort_count;
        let sort_slice = &mut dsr.entities[(curr_sort_count as usize)..];
        let to_sort_count = sort_slice.len().min(Self::DISTANCE_SORT_PER_UPDATE);

        if to_sort_count > 0 {
            sort_slice.select_nth_unstable_by(to_sort_count - 1, |&a, &b| {
                let a_transform = transform_comps.get(a);
                let a_mesh = vertex_mesh_comps.get(a);
                let b_transform = transform_comps.get(b);
                let b_mesh = vertex_mesh_comps.get(b);

                if a_transform.is_none() || a_mesh.is_none() || b_transform.is_none() || b_mesh.is_none() {
                    return cmp::Ordering::Equal;
                }

                let a_transform = a_transform.unwrap();
                let a_mesh = a_mesh.unwrap();
                let b_transform = b_transform.unwrap();
                let b_mesh = b_mesh.unwrap();

                let a_pos = {
                    let aabb = *a_mesh.0.aabb();
                    (aabb.0 + aabb.1) * 0.5 + a_transform.position
                };
                let b_pos = {
                    let aabb = *b_mesh.0.aabb();
                    (aabb.0 + aabb.1) * 0.5 + b_transform.position
                };

                let a_dist = (a_pos - self.camera_pos).magnitude();
                let b_dist = (b_pos - self.camera_pos).magnitude();

                if a_dist < b_dist {
                    cmp::Ordering::Less
                } else if a_dist > b_dist {
                    cmp::Ordering::Greater
                } else {
                    cmp::Ordering::Equal
                }
            });
        }

        dsr.curr_sort_count += to_sort_count as u32;
        if dsr.curr_sort_count >= dsr.entities.len() as u32 {
            dsr.curr_sort_count = 0;
        }
    }
}

// Updates model transform matrices
pub(super) struct TransformEventsSystem {
    pub transform_comps: Arc<RwLock<scene::ComponentStorage<component::Transform>>>,
    pub model_transform_comps: Arc<RwLock<scene::ComponentStorage<component::ModelTransform>>>,
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
                    Self::transform_modified(
                        transform_comps.get(entity).unwrap(),
                        model_transform_comps.get_mut(entity).unwrap(),
                    );
                }
                Event::Modified(entity) => {
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
    pub world_transform_comps: Arc<RwLock<scene::ComponentStorage<component::WorldTransform>>>,
    pub renderer_comps: Arc<RwLock<scene::ComponentStorage<component::Renderer>>>,
}

impl WorldTransformEventsSystem {
    fn world_transform_modified(
        world_transform: &component::WorldTransform,
        renderer: &component::Renderer,
        buffer_updates: &mut Vec<BufferUpdate>,
    ) {
        let matrix_bytes = unsafe {
            slice::from_raw_parts(
                world_transform.matrix.as_ptr() as *const u8,
                mem::size_of::<na::Matrix4<f32>>(),
            )
            .to_vec()
        };

        buffer_updates.push(BufferUpdate {
            buffer: Arc::clone(&renderer.uniform_buffer),
            offset: renderer.mat_pipeline.uniform_buffer_offset_model() as u64,
            data: matrix_bytes,
        });
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
                        renderer_comps.get(entity).unwrap(),
                        &mut buffer_updates,
                    );
                }
                Event::Modified(entity) => {
                    Self::world_transform_modified(
                        world_transform_comps.get(entity).unwrap(),
                        renderer_comps.get(entity).unwrap(),
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
    pub parent_comps: Arc<RwLock<scene::ComponentStorage<component::Parent>>>,
    pub children_comps: Arc<RwLock<scene::ComponentStorage<component::Children>>>,
    pub model_transform_comps: Arc<RwLock<scene::ComponentStorage<component::ModelTransform>>>,
    pub world_transform_comps: Arc<RwLock<scene::ComponentStorage<component::WorldTransform>>>,
}

impl HierarchyPropagationSystem {
    fn propagate_hierarchy(
        parent_world_transform: component::WorldTransform,
        parent_world_transform_changed: bool,
        parent_entity: u32,
        entity: u32,
        parent_comps: &mut scene::ComponentStorage<component::Parent>,
        children_comps: &scene::ComponentStorage<component::Children>,
        model_transform_comps: &mut scene::ComponentStorage<component::ModelTransform>,
        world_transform_comps: &mut scene::ComponentStorage<component::WorldTransform>,
    ) {
        let model_transform = model_transform_comps.get_mut_unchecked(entity).unwrap();

        let new_world_transform = parent_world_transform.combine(model_transform);
        let world_transform_changed = parent_world_transform_changed || model_transform.changed;

        if model_transform.changed {
            model_transform.changed = false;
        }
        if world_transform_changed {
            *world_transform_comps.get_mut(entity).unwrap() = new_world_transform;
        }

        *parent_comps.get_mut_unchecked(entity).unwrap() = component::Parent(parent_entity);

        if let Some(children) = children_comps.get(entity) {
            for child in &children.0 {
                Self::propagate_hierarchy(
                    new_world_transform,
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
            .alive_entries()
            .difference(parent_comps.alive_entries())
            .collect();

        for entity in entities {
            let (model_transform_changed, world_transform) = {
                let model_transform = model_transform_comps.get_mut_unchecked(entity as u32).unwrap();
                let world_transform = component::WorldTransform::from_model_transform(&model_transform);
                let model_transform_changed = model_transform.changed;

                if model_transform_changed {
                    *world_transform_comps.get_mut(entity as u32).unwrap() = world_transform;
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
