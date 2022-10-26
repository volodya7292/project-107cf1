use std::sync::Arc;

use crate::renderer;

#[derive(Default)]
pub struct VertexMesh(pub(crate) Arc<renderer::vertex_mesh::RawVertexMesh>);

impl VertexMesh {
    pub fn new(vertex_mesh: &Arc<renderer::vertex_mesh::RawVertexMesh>) -> VertexMesh {
        VertexMesh(Arc::clone(vertex_mesh))
    }
}
