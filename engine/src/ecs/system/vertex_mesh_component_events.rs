use std::sync::Arc;
use std::time::Instant;

use entity_data::{EntityId, SystemAccess, SystemHandler};

use crate::ecs::component;
use crate::renderer::vertex_mesh::RawVertexMesh;
use crate::utils::{HashMap, HashSet};

pub(crate) struct VertexMeshCompEvents<'a> {
    pub vertex_meshes: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub dirty_components: HashSet<EntityId>,
    pub buffer_updates: &'a mut HashMap<EntityId, Arc<RawVertexMesh>>,
    pub run_time: f64,
}

impl SystemHandler for VertexMeshCompEvents<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let vertex_mesh_comps = data.component_mut::<component::VertexMesh>();

        // Update device buffers of vertex meshes
        for entity in &self.dirty_components {
            if let Some(comp) = vertex_mesh_comps.get(entity) {
                self.buffer_updates.insert(*entity, Arc::clone(&comp.0));
            }
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
