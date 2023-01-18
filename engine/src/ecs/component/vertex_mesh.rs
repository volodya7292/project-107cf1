use crate::renderer::vertex_mesh::RawVertexMesh;
use std::sync::Arc;

#[derive(Default)]
pub struct VertexMesh(pub(crate) Arc<RawVertexMesh>);

impl VertexMesh {
    pub fn new(vertex_mesh: &Arc<RawVertexMesh>) -> VertexMesh {
        VertexMesh(Arc::clone(vertex_mesh))
    }

    pub fn without_data(vertices: u32, instances: u32) -> VertexMesh {
        VertexMesh(RawVertexMesh::without_data(vertices, instances))
    }
}
