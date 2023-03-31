use crate::module::main_renderer::camera::PerspectiveCamera;
use crate::module::main_renderer::material_pipeline::MaterialPipelineSet;
use crate::module::main_renderer::resources::Renderable;
use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use common::glm::Vec3;
use common::types::HashMap;
use entity_data::{EntityId, EntityStorage};
use std::sync::Arc;
use vk_wrapper::{DescriptorSet, DeviceBuffer};

pub mod depth;

pub struct StageContext<'a> {
    pub storage: &'a EntityStorage,
    pub material_pipelines: &'a [MaterialPipelineSet],
    pub ordered_entities: &'a [EntityId],
    pub active_camera: &'a PerspectiveCamera,
    pub relative_camera_pos: Vec3,
    pub curr_vertex_meshes: &'a HashMap<EntityId, Arc<RawVertexMesh>>,
    pub renderables: &'a HashMap<EntityId, Renderable>,
    pub g_per_frame_in: DescriptorSet,
    pub per_frame_ub: &'a DeviceBuffer,
    pub uniform_buffer_basic: &'a DeviceBuffer,
    pub render_size: (u32, u32),
}
