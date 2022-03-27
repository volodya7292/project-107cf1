use crate::component;
use crate::render_engine::scene;
use crate::render_engine::scene::{ComponentStorageImpl, Entity, Event};
use crate::render_engine::vertex_mesh::RawVertexMesh;
use crate::utils::{HashMap, LruCache};
use std::sync::Arc;

pub struct VertexMeshCompEvents<'a> {
    pub vertex_meshes: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub vertex_mesh_comps: scene::LockedStorage<'a, component::VertexMesh>,
    pub buffer_updates: &'a mut LruCache<Entity, Arc<RawVertexMesh>>,
}

impl VertexMeshCompEvents<'_> {
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
