use crate::ecs::component;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity, Event, LockedStorage};
use crate::renderer::vertex_mesh::RawVertexMesh;
use crate::renderer::GVBVertexMesh;
use crate::utils::HashMap;
use std::collections::hash_map;
use std::sync::Arc;

pub(crate) struct VertexMeshCompEvents<'a> {
    pub gvb_vertex_meshes: &'a mut HashMap<usize, GVBVertexMesh>,
    pub entity_vertex_meshes: &'a mut HashMap<Entity, usize>,
    pub vertex_mesh_comps: LockedStorage<'a, component::VertexMesh>,
    pub vertex_mesh_updates: &'a mut HashMap<Entity, Arc<RawVertexMesh>>,
    pub to_remove_vertex_meshes: &'a mut HashMap<usize, u32>,
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
                    let vm = vertex_mesh_comps.get(*entity).unwrap();

                    if let Some(staging_buffer) = &vm.0.staging_buffer {
                        let mesh_ptr = staging_buffer.as_ptr() as usize;

                        match self.gvb_vertex_meshes.entry(mesh_ptr) {
                            hash_map::Entry::Occupied(mut e) => {
                                e.get_mut().ref_count += 1;
                            }
                            hash_map::Entry::Vacant(_) => {
                                self.vertex_mesh_updates.insert(*entity, Arc::clone(&vm.0));
                            }
                        }
                    } else {
                        // If new mesh is empty, remove previous vertex mesh if it is assigned
                        if let Some(prev_mesh_ptr) = self.entity_vertex_meshes.remove(entity) {
                            let remove_refs_n =
                                self.to_remove_vertex_meshes.entry(prev_mesh_ptr).or_insert(0);
                            *remove_refs_n += 1;
                        }
                    };
                }
                Event::Removed(entity) => {
                    if let Some(mesh_ptr) = self.entity_vertex_meshes.remove(entity) {
                        let remove_refs_n = self.to_remove_vertex_meshes.entry(mesh_ptr).or_insert(0);
                        *remove_refs_n += 1;
                    }
                }
            }
        }
    }
}
