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
            position: Vector3::new(0_f32, 0_f32, 0_f32),
            rotation: Vector3::new(0_f32, 0_f32, 0_f32),
            scale: Vector3::new(1_f32, 1_f32, 1_f32),
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

    uniform_buffer: Arc<vkw::DeviceBuffer>,

    //buffers: HashMap<u32, vkw::RawHostBuffer>,
    // binding id -> renderer impl-specific res index
    translucent: bool,
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
            uniform_buffer: device
                .create_device_buffer(
                    vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::UNIFORM,
                    0,
                    1, // TODO
                )
                .unwrap(),
            translucent,
            changed: true,
        }
    }
}

impl specs::Component for Renderer {
    type Storage = specs::VecStorage<Self>;
}
