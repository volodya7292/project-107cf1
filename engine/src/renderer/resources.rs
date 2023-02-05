use crate::renderer::material_pipeline::MaterialPipelineSet;
use crate::renderer::texture_atlas::TextureAtlas;
use crate::renderer::vertex_mesh::RawVertexMesh;
use crate::renderer::MaterialInfo;
use base::utils::HashMap;
use entity_data::EntityId;
use index_pool::IndexPool;
use nalgebra_glm::Vec4;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::sync::Arc;
use vk_wrapper::{
    CmdList, DescriptorPool, DescriptorSet, DeviceBuffer, Framebuffer, HostBuffer, Image, ImageView,
    Pipeline, PipelineSignature, RenderPass, Sampler, Shader, SubmitPacket,
};

pub(crate) const N_CUSTOM_DESCRIPTORS: usize = 1;
pub(crate) const CUSTOM_OBJECT_DESCRIPTOR_IDX: usize = 0;

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
    pub vertex_meshes: HashMap<EntityId, Arc<RawVertexMesh>>,
    pub material_pipelines: Vec<MaterialPipelineSet>,

    // Temporary resources
    pub renderables_to_destroy: Vec<Renderable>,
    pub vertex_meshes_to_destroy: Vec<Arc<RawVertexMesh>>,
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
