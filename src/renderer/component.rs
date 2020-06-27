use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::vertex_mesh::RawVertexMesh;
use nalgebra::Vector3;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;

pub struct Transform {
    position: Vector3<f32>,
    rotation: Vector3<f32>,
    scale: Vector3<f32>,
    changed: bool,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            changed: true,
        }
    }
}

impl specs::Component for Transform {
    type Storage = specs::VecStorage<Self>;
}

pub struct VertexMeshRef {
    vertex_mesh: Arc<Mutex<RawVertexMesh>>,
    changed: bool,
}

impl VertexMeshRef {
    pub fn new(vertex_mesh: Arc<Mutex<RawVertexMesh>>) -> VertexMeshRef {
        Self {
            vertex_mesh,
            changed: true,
        }
    }
}

impl specs::Component for VertexMeshRef {
    type Storage = specs::VecStorage<Self>;
}

pub struct Renderer {
    mat_pipeline: Arc<MaterialPipeline>,
    uniform_buffer_offset: u64,

    //buffers: HashMap<u32, vkw::RawHostBuffer>,
    // binding id -> renderer impl-specific res index
    translucent: bool,
    visible_on_camera: bool,
    changed: bool,
}

impl Renderer {
    pub fn new(
        device: &Arc<vkw::Device>,
        mat_pipeline: Arc<MaterialPipeline>,
        translucent: bool,
    ) -> Renderer {
        Self {
            mat_pipeline,
            uniform_buffer_offset: 0,
            translucent,
            visible_on_camera: false,
            changed: true,
        }
    }
}

impl specs::Component for Renderer {
    type Storage = specs::VecStorage<Self>;
}
