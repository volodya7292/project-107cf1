use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::vertex_mesh::VertexMesh;
use nalgebra::Vector3;
use std::collections::HashMap;
use std::sync::Arc;
use vk_wrapper as vkw;

pub struct Transform {
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
}

impl Transform {
    pub fn new() -> Transform {
        Transform {
            position: Vector3::new(0_f32, 0_f32, 0_f32),
            rotation: Vector3::new(0_f32, 0_f32, 0_f32),
            scale: Vector3::new(1_f32, 1_f32, 1_f32),
        }
    }
}

impl specs::Component for Transform {
    type Storage = specs::VecStorage<Self>;
}

struct VertexMeshRef(Arc<VertexMesh>);

impl specs::Component for VertexMeshRef {
    type Storage = specs::VecStorage<Self>;
}

pub struct Renderer {
    mat_pipeline: Arc<MaterialPipeline>,
    buffers: HashMap<u32, vkw::RawHostBuffer>,
    // binding id -> renderer impl-specific res index
    translucent: bool,
}

impl specs::Component for Renderer {
    type Storage = specs::VecStorage<Self>;
}
