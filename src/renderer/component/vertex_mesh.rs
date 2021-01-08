use crate::renderer;
use std::sync::Arc;

pub struct VertexMesh(pub(in crate::renderer) Arc<renderer::vertex_mesh::RawVertexMesh>);

impl VertexMesh {
    pub fn new(vertex_mesh: &Arc<renderer::vertex_mesh::RawVertexMesh>) -> VertexMesh {
        VertexMesh(Arc::clone(vertex_mesh))
    }
}
