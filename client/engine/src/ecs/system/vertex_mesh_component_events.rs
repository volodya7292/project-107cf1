use crate::ecs::component::VertexMeshC;
use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use crate::module::scene::change_manager::{ChangeType, ComponentChange};
use common::types::HashMap;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use std::sync::Arc;
use std::time::Instant;

pub(crate) struct VertexMeshCompEvents<'a> {
    pub component_changes: Vec<ComponentChange>,
    pub curr_vertex_meshes: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub completed_updates: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub new_buffer_updates: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub run_time: f64,
}

impl SystemHandler for VertexMeshCompEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let vertex_mesh_comps = data.component_mut::<VertexMeshC>();

        // Update device buffers of vertex meshes
        for change in &self.component_changes {
            let entity = change.entity();

            if change.ty() == ChangeType::Removed {
                self.curr_vertex_meshes.remove(entity);
                self.completed_updates.remove(entity);
                // New vertex mesh doesn't need to be uploaded to the GPU
                self.new_buffer_updates.remove(entity);
                continue;
            }

            if let Some(comp) = vertex_mesh_comps.get(entity) {
                self.new_buffer_updates.insert(*entity, Arc::clone(&comp.0));
            }
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
