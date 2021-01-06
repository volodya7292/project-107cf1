use crate::renderer::{component, DistanceSortedRenderables};
use nalgebra as na;
use specs::storage::ComponentEvent;
use specs::{Join, System};
use std::cmp;
use std::sync::atomic::AtomicU32;
use std::sync::{Arc, Mutex};

pub(super) struct RendererCompEventsSystem {
    pub renderer_comp_reader: Arc<Mutex<specs::ReaderId<specs::storage::ComponentEvent>>>,
    pub sorted_renderables: Arc<Mutex<DistanceSortedRenderables>>,
}

impl<'a> specs::System<'a> for RendererCompEventsSystem {
    type SystemData = (specs::Entities<'a>, specs::ReadStorage<'a, component::Renderer>);

    fn run(&mut self, (entities, renderer_comps): Self::SystemData) {
        let mut dsr = self.sorted_renderables.lock().unwrap();
        let mut sorted_renderables = &mut dsr.entities;
        let mut removed_count = 1;

        // Add new objects to sort
        // -------------------------------------------------------------------------------------------------------------
        {
            let renderer_comp_events = renderer_comps
                .channel()
                .read(&mut self.renderer_comp_reader.lock().unwrap());

            let mut inserted = specs::BitSet::new();

            for event in renderer_comp_events {
                match event {
                    ComponentEvent::Inserted(i) => {
                        inserted.add(*i);
                    }
                    ComponentEvent::Removed(_) => {
                        removed_count += 1;
                    }
                    _ => {}
                }
            }

            for (entity, _comp, _) in (&entities, &renderer_comps, &inserted).join() {
                sorted_renderables.push(entity);
            }
        }

        // Replace removed(dead) entities with alive ones
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut swap_entities = Vec::<specs::Entity>::with_capacity(removed_count);
            let mut new_len = sorted_renderables.len();

            // Find alive entities for replacement
            for &entity in sorted_renderables.iter().rev() {
                if entities.is_alive(entity) {
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
                if !entities.is_alive(*entity) {
                    *entity = swap_entities.remove(swap_entities.len() - 1);
                }
            }

            // Add the rest of swap_entities that were not swapped due to resized vector
            sorted_renderables.extend(swap_entities);
        }
    }
}

// Sort render objects from front to back (for Z rejection & occlusion queries)
pub(super) struct DistanceSortSystem {
    pub sorted_renderables: Arc<Mutex<DistanceSortedRenderables>>,
    pub camera_pos: na::Vector3<f32>,
}

impl DistanceSortSystem {
    const DISTANCE_SORT_PER_UPDATE: usize = 128;
}

impl<'a> specs::System<'a> for DistanceSortSystem {
    type SystemData = (
        specs::ReadStorage<'a, component::Transform>,
        specs::ReadStorage<'a, component::VertexMesh>,
    );

    fn run(&mut self, (transform_comps, vertex_mesh_comps): Self::SystemData) {
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
