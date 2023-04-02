use crate::module::main_renderer::material_pipeline::MaterialPipelineSet;
use crate::module::main_renderer::texture_atlas::TextureAtlas;
use crate::module::main_renderer::vertex_mesh::RawVertexMesh;
use common::glm::Vec4;
use common::types::HashMap;
use entity_data::EntityId;
use index_pool::IndexPool;
use smallvec::SmallVec;
use std::sync::Arc;
use vk_wrapper::{
    DescriptorPool, DescriptorSet, DeviceBuffer, Framebuffer, HostBuffer, Image, ImageView, Pipeline,
    PipelineSignature, PrimitiveTopology, RenderPass, Sampler, Shader,
};

pub(crate) const N_CUSTOM_DESCRIPTORS: usize = 1;
pub(crate) const GENERAL_OBJECT_DESCRIPTOR_IDX: usize = 0;

pub struct RendererResources {
    pub texture_atlases: [TextureAtlas; 4],
    pub _tex_atlas_sampler: Arc<Sampler>,

    pub depth_render_pass: Arc<RenderPass>,
    pub depth_pyramid_image: Option<Arc<Image>>,
    pub depth_pyramid_views: Vec<Arc<ImageView>>,
    pub depth_framebuffer: Option<Arc<Framebuffer>>,
    pub depth_pyramid_pipeline: Arc<Pipeline>,
    pub depth_pyramid_signature: Arc<PipelineSignature>,
    pub depth_pyramid_pool: Option<DescriptorPool>,
    pub depth_pyramid_descs: Vec<DescriptorSet>,

    pub cull_pipeline: Arc<Pipeline>,
    pub cull_signature: Arc<PipelineSignature>,
    pub cull_pool: DescriptorPool,
    pub cull_desc: DescriptorSet,
    pub cull_buffer: DeviceBuffer,
    pub cull_host_buffer: HostBuffer<CullObject>,
    pub visibility_buffer: DeviceBuffer,
    pub visibility_host_buffer: HostBuffer<u32>,

    pub g_render_pass: Arc<RenderPass>,
    pub g_framebuffer: Option<Arc<Framebuffer>>,
    pub g_per_frame_pool: DescriptorPool,
    pub g_per_frame_in: DescriptorSet,

    pub translucency_depths_pixel_shader: Arc<Shader>,
    pub translucency_depths_image: Option<DeviceBuffer>,
    pub translucency_colors_image: Option<Arc<Image>>,

    pub per_frame_ub: DeviceBuffer,
    pub material_buffer: DeviceBuffer,

    pub uniform_buffer_basic: DeviceBuffer,
    pub uniform_buffer_offsets: IndexPool,

    pub renderables: HashMap<EntityId, Renderable>,
    /// Meshes ready to be used on GPU (their staging buffers have been uploaded to GPU)
    pub curr_vertex_meshes: HashMap<EntityId, Arc<RawVertexMesh>>,
    pub material_pipelines: Vec<MaterialPipelineSet>,
}

#[derive(Copy, Clone)]
#[repr(C)]
pub struct CullObject {
    pub sphere: Vec4,
    pub id: u32,
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