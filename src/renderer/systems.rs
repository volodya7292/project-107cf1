use crate::renderer::{component, scene, DistanceSortedRenderables};
use nalgebra as na;
use std::cmp;
use std::sync::{atomic, Arc, Mutex, RwLock};
use vk_wrapper as vkw;
use vk_wrapper::SubmitPacket;

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

pub(super) struct RendererCompEventsData {
    pub renderer_comps: Arc<RwLock<scene::ComponentStorage<component::Renderer>>>,
    pub sorted_renderables: Arc<Mutex<DistanceSortedRenderables>>,
    pub depth_per_object_pool: Arc<vkw::DescriptorPool>,
    pub model_inputs: Arc<vkw::DescriptorPool>,
}

impl RendererCompEventsData {
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
                        renderer_comp_modified(
                            renderer_comps.get_mut_unchecked(i).unwrap(),
                            &self.depth_per_object_pool,
                            &self.model_inputs,
                        );
                        sorted_renderables.push(i);
                    }
                    scene::Event::Modified(i) => {
                        renderer_comp_modified(
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

pub(super) struct VertexMeshCompEventsData {
    pub events: Vec<scene::Event>,
    pub vertex_mesh_comps: Arc<RwLock<scene::ComponentStorage<component::VertexMesh>>>,
    pub device: Arc<vkw::Device>,
    pub staging_cl: Arc<Mutex<vkw::CmdList>>,
    pub staging_submit: Arc<Mutex<SubmitPacket>>,
}

impl VertexMeshCompEventsData {
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
pub(super) struct DistanceSortData {
    pub transform_comps: Arc<RwLock<scene::ComponentStorage<component::Transform>>>,
    pub vertex_mesh_comps: Arc<RwLock<scene::ComponentStorage<component::VertexMesh>>>,
    pub sorted_renderables: Arc<Mutex<DistanceSortedRenderables>>,
    pub camera_pos: na::Vector3<f32>,
}

impl DistanceSortData {
    const DISTANCE_SORT_PER_UPDATE: usize = 128;

    pub fn run(&mut self) {
        let transform_comps = self.transform_comps.read().unwrap();
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
                    (aabb.0 + aabb.1) * 0.5 + a_transform.position()
                };
                let b_pos = {
                    let aabb = *b_mesh.0.aabb();
                    (aabb.0 + aabb.1) * 0.5 + b_transform.position()
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
