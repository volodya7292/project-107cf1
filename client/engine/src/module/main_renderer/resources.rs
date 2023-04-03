use crate::module::main_renderer::material_pipeline::MaterialPipelineSet;
use crate::module::main_renderer::texture_atlas::TextureAtlas;
use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use common::types::HashMap;
use entity_data::EntityId;
use index_pool::IndexPool;
use smallvec::SmallVec;
use std::sync::Arc;
use vk_wrapper::{
    DescriptorPool, DescriptorSet, DeviceBuffer, PipelineSignature, PrimitiveTopology, Sampler, Shader,
};

pub(crate) const N_CUSTOM_DESCRIPTORS: usize = 1;
pub(crate) const GENERAL_OBJECT_DESCRIPTOR_IDX: usize = 0;

pub struct RendererResources {
    pub texture_atlases: [TextureAtlas; 4],
    pub _tex_atlas_sampler: Arc<Sampler>,

    pub g_per_frame_pool: DescriptorPool,
    pub g_per_frame_in: DescriptorSet,

    pub per_frame_ub: DeviceBuffer,
    pub material_buffer: DeviceBuffer,

    pub uniform_buffer_basic: DeviceBuffer,
    pub uniform_buffer_offsets: IndexPool,

    pub renderables: HashMap<EntityId, Renderable>,
    /// Meshes ready to be used on GPU (their staging buffers have been uploaded to GPU)
    pub curr_vertex_meshes: HashMap<EntityId, Arc<RawVertexMesh>>,
    pub material_pipelines: Vec<MaterialPipelineSet>,
}

pub struct Renderable {
    pub buffers: SmallVec<[DeviceBuffer; 4]>,
    pub mat_pipeline: u32,
    pub uniform_buf_index: usize,
    pub descriptor_sets: [DescriptorSet; N_CUSTOM_DESCRIPTORS],
}

pub struct MaterialPipelineParams<'a> {
    pub shaders: &'a [Arc<Shader>],
    pub topology: PrimitiveTopology,
    pub cull_back_faces: bool,
    pub main_signature: &'a Arc<PipelineSignature>,
}
