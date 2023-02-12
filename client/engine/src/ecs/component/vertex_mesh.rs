use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use std::sync::Arc;

#[derive(Default)]
pub struct VertexMeshC(pub(crate) Arc<RawVertexMesh>);

impl VertexMeshC {
    pub fn new(vertex_mesh: &Arc<RawVertexMesh>) -> VertexMeshC {
        VertexMeshC(Arc::clone(vertex_mesh))
    }

    pub fn without_data(vertices: u32, instances: u32) -> VertexMeshC {
        VertexMeshC(RawVertexMesh::without_data(vertices, instances))
    }
}
