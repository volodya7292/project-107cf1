//! ECS components. It is invalid to have multiple incompatible components in the same entity.
//! `MeshRenderConfig` and `SimpleText` are incompatible
pub(crate) mod internal;
pub mod render_config;
pub mod simple_text;
pub mod transform;
pub mod vertex_mesh;

pub use render_config::MeshRenderConfig;
pub use simple_text::SimpleText;
pub use transform::Transform;
pub use vertex_mesh::VertexMesh;
