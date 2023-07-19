use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use std::sync::Arc;

#[derive(Default)]
pub struct VertexMeshC {
    pub(crate) raw_mesh: Arc<RawVertexMesh>,
    pub(crate) load_immediate: bool,
}

impl VertexMeshC {
    pub fn new(vertex_mesh: &Arc<RawVertexMesh>) -> VertexMeshC {
        VertexMeshC {
            raw_mesh: Arc::clone(vertex_mesh),
            load_immediate: false,
        }
    }

    /// Specifies that the staging buffer should be loaded to GPU immediately before the next frame.
    pub fn with_load_immediate(mut self) -> Self {
        self.load_immediate = true;
        self
    }

    pub fn without_data(vertices: u32, instances: u32) -> VertexMeshC {
        VertexMeshC {
            raw_mesh: RawVertexMesh::without_data(vertices, instances),
            load_immediate: true,
        }
    }
}
