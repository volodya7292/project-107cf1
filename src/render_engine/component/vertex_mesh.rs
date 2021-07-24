use crate::render_engine;
use std::sync::Arc;

pub struct VertexMesh(pub(in crate::render_engine) Arc<render_engine::vertex_mesh::RawVertexMesh>);

impl VertexMesh {
    pub fn new(vertex_mesh: &Arc<render_engine::vertex_mesh::RawVertexMesh>) -> VertexMesh {
        VertexMesh(Arc::clone(vertex_mesh))
    }
}
