use std::sync::Arc;

use crate::ecs::component;
use crate::ecs::scene_storage;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity, Event};
use crate::renderer::vertex_mesh::RawVertexMesh;
use crate::utils::HashMap;

pub(crate) struct VertexMeshCompEvents<'a> {
    pub vertex_meshes: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene_storage::LockedStorage<'a, component::VertexMesh>,
    pub buffer_updates: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub removed_entities: &'a mut Vec<Entity>,
}

impl VertexMeshCompEvents<'_> {
    pub fn run(&mut self) {
        let events = self.vertex_mesh_comps.write().events();
        let vertex_mesh_comps = self.vertex_mesh_comps.read();

        // Update device buffers of vertex meshes
        // ------------------------------------------------------------------------------------
        for event in &events {
            match event {
                Event::Created(entity) | Event::Modified(entity) => {
                    let comp = vertex_mesh_comps.get(*entity).unwrap();
                    self.buffer_updates.insert(*entity, Arc::clone(&comp.0));
                }
                Event::Removed(entity) => {
                    self.vertex_meshes.remove(entity);
                    self.buffer_updates.remove(entity);
                    self.removed_entities.push(*entity);
                }
            }
        }
    }
}
