use crate::renderer::vertex_mesh::RawVertexMesh;
use std::sync::{Arc, Mutex};

pub struct VertexMeshRef {
    pub vertex_mesh: Arc<Mutex<RawVertexMesh>>,
    changed: bool,
}

impl VertexMeshRef {
    pub fn new(vertex_mesh: &Arc<Mutex<RawVertexMesh>>) -> VertexMeshRef {
        Self {
            vertex_mesh: Arc::clone(vertex_mesh),
            changed: true,
        }
    }
}

impl specs::Component for VertexMeshRef {
    type Storage = specs::VecStorage<Self>;
}
