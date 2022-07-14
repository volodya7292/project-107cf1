//! ## Overall rendering pipeline
//! 1. Depth pass  
//! 1.1. Render depth image of solid objects.  
//! 1.2. Construct depth pyramid.  
//! 1.3. Use depth pyramid to cull objects by occlusion.  
//! 1.4. Render translucent objects: find `TRANSLUCENCY_DEPTH_LAYERS` closest depths.
//! 4. Render solid objects into g-buffer (with early-z testing).  
//! 5. Render translucent objects: match depths with respective colors.
//! 6. Compose solid and translucent colors.

// Notes
// --------------------------------------------
// Encountered causes of VK_ERROR_DEVICE_LOST:
// - Out of bounds access:
//   - Incorrect indices of vertex mesh.
//
// Swapchain creation error cause may be *out of device memory*.
//
// HLSL `globallycoherent` and GLSL `coherent` modifiers do not work with MoltenVK (Metal).
//

mod texture_atlas;

#[macro_use]
pub mod material_pipeline;
#[macro_use]
pub mod vertex_mesh;
pub mod camera;
mod helpers;
pub mod module;

use crate::ecs::component::internal::{Children, GlobalTransform, Parent};
use crate::ecs::scene::Scene;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity};
use crate::ecs::{component, system};
use crate::renderer::module::RendererModule;
use crate::utils::HashMap;
use crate::{unwrap_option, utils};
use basis_universal::TranscoderTextureFormat;
use camera::{Camera, Frustum};
use index_pool::IndexPool;
use lazy_static::lazy_static;
use material_pipeline::MaterialPipelineSet;
pub use module::text_renderer::FontSet;
use nalgebra::{Matrix4, Vector4};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, U32Vec4, UVec2, Vec2, Vec3, Vec4};
use parking_lot::Mutex;
use rayon::prelude::*;
use smallvec::{smallvec, SmallVec, ToSmallVec};
use std::any::TypeId;
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::Instant;
use std::{fs, iter, mem, slice};
use texture_atlas::TextureAtlas;
use vertex_mesh::RawVertexMesh;
pub use vertex_mesh::VertexMesh;
use vertex_mesh::VertexMeshCmdList;
use vk_wrapper::buffer::{BufferHandle, BufferHandleImpl};
use vk_wrapper::sampler::SamplerClamp;
use vk_wrapper::{
    swapchain, AccessFlags, Attachment, AttachmentRef, BindingLoc, BindingRes, BindingType, BufferUsageFlags,
    ClearValue, CmdList, CopyRegion, DescriptorPool, DescriptorSet, Device, DeviceBuffer, Format,
    Framebuffer, HostBuffer, Image, ImageLayout, ImageMod, ImageUsageFlags, ImageView, LoadStore, Pipeline,
    PipelineSignature, PipelineStageFlags, PrimitiveTopology, Queue, RenderPass, Sampler, SamplerFilter,
    SamplerMipmap, Shader, ShaderBinding, ShaderStageFlags, SignalSemaphore, SubmitInfo, SubmitPacket,
    Subpass, Surface, Swapchain, SwapchainImage, WaitSemaphore,
};

// TODO: Defragment VK memory (every frame?).
// TODO: Relocate memory from CPU (that was allocated there due to out of device-local memory) onto GPU.

#[derive(Default, Copy, Clone)]
pub struct UpdateTimings {
    pub systems_batch0: f64,
    pub batch0_render_events: f64,
    pub batch0_vertex_meshes: f64,
    pub batch0_hierarchy_propag: f64,
    pub systems_batch1: f64,
    pub batch1_global_transforms: f64,
    pub batch1_buffer_updates: f64,
    pub batch1_updates_commit: f64,
    pub uniform_buffers_update: f64,
    pub total: f64,
}

#[derive(Default, Copy, Clone)]
pub struct RenderTimings {
    pub depth_pass: f64,
    pub color_pass: f64,
    pub total: f64,
}

#[derive(Default, Copy, Clone)]
pub struct RendererTimings {
    pub update: UpdateTimings,
    pub render: RenderTimings,
}

impl Display for RendererTimings {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "batch0 {:.5} | batch1 {:.5} | uniforms_update {:.5} || \
            depth_pass {:.5} | color_pass {:.5} || upd_total {:.5} | render_total {:.5}",
            self.update.systems_batch0,
            self.update.systems_batch1,
            self.update.uniform_buffers_update,
            self.render.depth_pass,
            self.render.color_pass,
            self.update.total,
            self.render.total
        ))
    }
}

pub struct Renderer {
    scene: Scene,
    active_camera: Camera,
    camera_pos_pivot: DVec3,
    relative_camera_pos: DVec3,

    surface: Arc<Surface>,
    swapchain: Option<Swapchain>,
    surface_changed: bool,
    surface_size: (u32, u32),
    settings: Settings,
    device: Arc<Device>,

    texture_atlases: [TextureAtlas; 4],
    tex_atlas_sampler: Arc<Sampler>,

    staging_buffer: HostBuffer<u8>,
    transfer_cl: [Arc<Mutex<CmdList>>; 2],
    transfer_submit: [SubmitPacket; 2],
    staging_cl: Arc<Mutex<CmdList>>,
    staging_submit: SubmitPacket,
    modules_updates_submit: SubmitPacket,
    final_cl: [Arc<Mutex<CmdList>>; 2],
    final_submit: [SubmitPacket; 2],

    sw_framebuffers: Vec<Arc<Framebuffer>>,

    depth_secondary_cls: Vec<Arc<Mutex<CmdList>>>,
    translucency_depths_secondary_cls: Vec<Arc<Mutex<CmdList>>>,
    g_solid_secondary_cls: Vec<Arc<Mutex<CmdList>>>,
    g_translucent_secondary_cls: Vec<Arc<Mutex<CmdList>>>,

    depth_render_pass: Arc<RenderPass>,
    depth_pyramid_image: Option<Arc<Image>>,
    depth_pyramid_views: Vec<Arc<ImageView>>,
    depth_framebuffer: Option<Arc<Framebuffer>>,
    depth_pyramid_pipeline: Arc<Pipeline>,
    depth_pyramid_signature: Arc<PipelineSignature>,
    depth_pyramid_pool: Option<DescriptorPool>,
    depth_pyramid_descs: Vec<DescriptorSet>,

    cull_pipeline: Arc<Pipeline>,
    cull_signature: Arc<PipelineSignature>,
    cull_pool: DescriptorPool,
    cull_desc: DescriptorSet,
    cull_buffer: DeviceBuffer,
    cull_host_buffer: HostBuffer<CullObject>,
    visibility_buffer: DeviceBuffer,
    visibility_host_buffer: HostBuffer<u32>,

    sw_render_pass: Option<Arc<RenderPass>>,
    compose_pipeline: Option<Arc<Pipeline>>,
    compose_signature: Arc<PipelineSignature>,
    compose_pool: DescriptorPool,
    compose_desc: DescriptorSet,

    g_render_pass: Arc<RenderPass>,
    g_framebuffer: Option<Arc<Framebuffer>>,
    g_per_frame_pool: DescriptorPool,
    g_dyn_pool: DescriptorPool,
    g_per_frame_in: DescriptorSet,
    g_dyn_in: DescriptorSet,

    translucency_depths_pixel_shader: Arc<Shader>,
    translucency_depths_image: Option<DeviceBuffer>,
    translucency_colors_image: Option<Arc<Image>>,

    per_frame_ub: DeviceBuffer,
    material_buffer: DeviceBuffer,
    material_updates: HashMap<u32, MaterialInfo>,
    vertex_mesh_updates: HashMap<Entity, Arc<RawVertexMesh>>,
    vertex_mesh_pending_updates: Vec<VMBufferUpdate>,

    /// Entities ordered in respect to children order inside `Children` components:
    /// global parents are not in order, but all the children are.
    ordered_entities: Vec<Entity>,
    renderables: HashMap<Entity, Renderable>,
    vertex_meshes: HashMap<Entity, Arc<RawVertexMesh>>,
    pub(crate) material_pipelines: Vec<MaterialPipelineSet>,
    uniform_buffer_basic: DeviceBuffer,
    // device_buffers: SlotVec<DeviceBuffer>,
    uniform_buffer_offsets: IndexPool,

    modules: HashMap<TypeId, Box<dyn RendererModule + Send + Sync>>,
}

#[derive(Copy, Clone)]
pub enum TextureQuality {
    STANDARD = 128,
    HIGH = 256,
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum FPSLimit {
    VSync,
    Limit(u32),
}

#[derive(Copy, Clone)]
pub struct Settings {
    pub fps_limit: FPSLimit,
    pub prefer_triple_buffering: bool,
    pub textures_mipmaps: bool,
    pub texture_quality: TextureQuality,
    pub textures_max_anisotropy: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            fps_limit: FPSLimit::VSync,
            prefer_triple_buffering: true,
            textures_mipmaps: true,
            texture_quality: TextureQuality::STANDARD,
            textures_max_anisotropy: 1.0,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TextureAtlasType {
    ALBEDO = 0,
    SPECULAR = 1,
    EMISSION = 2,
    NORMAL = 3,
}

impl TextureAtlasType {
    fn basis_decode_type(&self) -> TranscoderTextureFormat {
        match self {
            TextureAtlasType::ALBEDO | TextureAtlasType::SPECULAR | TextureAtlasType::EMISSION => {
                TranscoderTextureFormat::BC7_RGBA
            }
            TextureAtlasType::NORMAL => TranscoderTextureFormat::BC5_RG,
        }
    }
}

#[derive(Debug)]
#[repr(C)]
struct CameraInfo {
    pos: Vector4<f32>,
    dir: Vector4<f32>,
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    z_near: f32,
    fovy: f32,
}

#[derive(Debug)]
#[repr(C)]
struct FrameInfoUniforms {
    camera: CameraInfo,
    atlas_info: U32Vec4,
    frame_size: UVec2,
}

#[repr(C)]
struct TranslucencyPushConsts {
    frame_size: UVec2,
}

#[repr(C)]
struct GPassConsts {
    is_translucent_pass: u32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct MaterialInfo {
    pub(crate) diffuse_tex_id: u32,
    pub(crate) specular_tex_id: u32,
    pub(crate) normal_tex_id: u32,
    pub(crate) diffuse: Vec4,
    pub(crate) specular: Vec4,
    pub(crate) emission: Vec4,
}

pub enum MatComponent {
    Texture(u16),
    Color(Vec4),
}

impl MaterialInfo {
    pub fn new(
        diffuse: MatComponent,
        specular: MatComponent,
        normal_tex_id: u16,
        emission: Vec4,
    ) -> MaterialInfo {
        let mut info = MaterialInfo {
            diffuse_tex_id: 0,
            specular_tex_id: 0,
            normal_tex_id: normal_tex_id as u32,
            diffuse: Default::default(),
            specular: Default::default(),
            emission,
        };

        match diffuse {
            MatComponent::Texture(id) => info.diffuse_tex_id = id as u32,
            MatComponent::Color(col) => info.diffuse = col,
        }
        match specular {
            MatComponent::Texture(id) => info.specular_tex_id = id as u32,
            MatComponent::Color(col) => info.specular = col,
        }

        info
    }
}

#[repr(C)]
struct DepthPyramidConstants {
    out_size: Vec2,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct CullObject {
    sphere: Vec4,
    id: u32,
}

#[repr(C)]
struct CullConstants {
    pyramid_size: Vec2,
    max_pyramid_levels: u32,
    object_count: u32,
}

pub(crate) struct Renderable {
    pub buffers: SmallVec<[DeviceBuffer; 4]>,
    pub mat_pipeline: u32,
    pub uniform_buf_index: usize,
    pub descriptor_sets: SmallVec<[DescriptorSet; 4]>,
}

pub(crate) struct BufferUpdate1 {
    pub buffer: BufferHandle,
    pub offset: u64,
    pub data: SmallVec<[u8; 256]>,
}

pub(crate) struct BufferUpdate2 {
    pub buffer: BufferHandle,
    pub data: SmallVec<[u8; 256]>,
    pub regions: Vec<CopyRegion>,
}

pub(crate) enum BufferUpdate {
    Type1(BufferUpdate1),
    Type2(BufferUpdate2),
}

pub(crate) struct VMBufferUpdate {
    pub entity: Entity,
    pub mesh: Arc<RawVertexMesh>,
}

pub const TEXTURE_ID_NONE: u16 = u16::MAX;

pub const MAX_OBJECT_COUNT: u32 = 65535;
pub const MAX_MATERIAL_COUNT: u32 = 4096;
pub const COMPUTE_LOCAL_THREADS: u32 = 32;
pub const MAX_BASIC_UNIFORM_BLOCK_SIZE: u64 = 256;

/// General descriptor set for engine-related resources
pub const DESC_SET_GENERAL_PER_FRAME: u32 = 0;
/// Descriptor set for custom per-object data (model matrix is mandatory)
pub const DESC_SET_CUSTOM_PER_OBJECT: u32 = 1;
/// Specific to material pipeline descriptor set for its per-frame resources
pub const DESC_SET_CUSTOM_PER_FRAME: u32 = 2;

const TRANSLUCENCY_N_DEPTH_LAYERS: u32 = 4;
const RESET_CAMERA_POS_THRESHOLD: f64 = 4096.0;

const PIPELINE_DEPTH_WRITE: u32 = 0;
const PIPELINE_TRANSLUCENCY_DEPTHS: u32 = 1;
const PIPELINE_COLOR: u32 = 2;
const PIPELINE_COLOR_WITH_BLENDING: u32 = 3;

lazy_static! {
    static ref PIPELINE_CACHE_FILENAME: &'static str = if cfg!(debug_assertions) {
        "pipeline_cache-debug"
    } else {
        "pipeline_cache"
    };

    static ref ADDITIONAL_PIPELINE_BINDINGS: [(BindingLoc, ShaderBinding); 8] = [
        // Per frame info
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 0),
            ShaderBinding {
                stage_flags: ShaderStageFlags::VERTEX | ShaderStageFlags::PIXEL,
                binding_type: BindingType::UNIFORM_BUFFER,
                count: 1,
            },
        ),
        // Per object info
        (
            BindingLoc::new(DESC_SET_CUSTOM_PER_OBJECT, 0),
            ShaderBinding {
                stage_flags: ShaderStageFlags::VERTEX,
                binding_type: BindingType::UNIFORM_BUFFER_DYNAMIC,
                count: 1,
            },
        ),
        // Material buffer
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 1),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::STORAGE_BUFFER,
                count: 1,
            },
        ),
        // Albedo atlas
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 2),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::SAMPLED_IMAGE,
                count: 1,
            },
        ),
        // Specular atlas
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 3),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::SAMPLED_IMAGE,
                count: 1,
            },
        ),
        // Normal atlas
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 4),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::SAMPLED_IMAGE,
                count: 1,
            },
        ),
        // Translucency depths (only used in translucency passes)
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 5),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::STORAGE_BUFFER,
                count: 1,
            },
        ),
        // Translucency colors (only used in translucency passes)
        (
            BindingLoc::new(DESC_SET_GENERAL_PER_FRAME, 6),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::STORAGE_IMAGE,
                count: 1,
            },
        ),
    ];
}

fn calc_group_count(thread_count: u32) -> u32 {
    (thread_count + COMPUTE_LOCAL_THREADS - 1) / COMPUTE_LOCAL_THREADS
}

impl Renderer {
    pub fn new(
        surface: &Arc<Surface>,
        size: (u32, u32),
        settings: Settings,
        device: &Arc<Device>,
        max_texture_count: u32,
    ) -> Renderer {
        // Load pipeline cache
        if let Ok(res) = fs::read(*PIPELINE_CACHE_FILENAME) {
            device.load_pipeline_cache(&res).unwrap();
        }

        let scene = Scene::new();
        let active_camera = Camera::new(1.0, std::f32::consts::FRAC_PI_2, 0.01);

        let transfer_queue = device.get_queue(Queue::TYPE_TRANSFER);
        let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);
        let present_queue = device.get_queue(Queue::TYPE_PRESENT);
        // Available threads in the render thread pool
        let available_threads = rayon::current_num_threads();

        let staging_buffer = device
            .create_host_buffer(BufferUsageFlags::TRANSFER_SRC, 0x800000)
            .unwrap();
        let per_frame_uniform_buffer = device
            .create_device_buffer(
                BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM,
                mem::size_of::<FrameInfoUniforms>() as u64,
                1,
            )
            .unwrap();
        let uniform_buffer_basic = device
            .create_device_buffer(
                BufferUsageFlags::UNIFORM | BufferUsageFlags::TRANSFER_DST,
                MAX_BASIC_UNIFORM_BLOCK_SIZE,
                MAX_OBJECT_COUNT as u64,
            )
            .unwrap();
        // TODO: allow different alignments
        assert_eq!(
            uniform_buffer_basic.aligned_element_size(),
            MAX_BASIC_UNIFORM_BLOCK_SIZE
        );
        // let uniform_buffer1 = device
        //     .create_device_buffer(
        //         BufferUsageFlags::UNIFORM | BufferUsageFlags::TRANSFER_DST,
        //         1024,
        //         MAX_OBJECT_COUNT as u64,
        //     )
        //     .unwrap();
        let material_buffer = device
            .create_device_buffer(
                BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE,
                mem::size_of::<MaterialInfo>() as u64,
                MAX_MATERIAL_COUNT as u64,
            )
            .unwrap();

        // Create depth pass resources
        // -----------------------------------------------------------------------------------------------------------------
        let depth_render_pass = device
            .create_render_pass(
                &[Attachment {
                    format: Format::D32_FLOAT,
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::SHADER_READ,
                    load_store: LoadStore::InitClearFinalSave,
                }],
                &[Subpass {
                    color: vec![],
                    depth: Some(AttachmentRef {
                        index: 0,
                        layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                    }),
                }],
                &[],
            )
            .unwrap();

        // Depth pyramid pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let depth_pyramid_compute = device
            .create_compute_shader(include_bytes!("../shaders/build/depth_pyramid.comp.spv"))
            .unwrap();
        let depth_pyramid_signature = device
            .create_pipeline_signature(&[depth_pyramid_compute], &[])
            .unwrap();
        let depth_pyramid_pipeline = device.create_compute_pipeline(&depth_pyramid_signature).unwrap();

        // Cull pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let cull_compute = device
            .create_compute_shader(include_bytes!("../shaders/build/cull.comp.spv"))
            .unwrap();
        let cull_signature = device.create_pipeline_signature(&[cull_compute], &[]).unwrap();
        let cull_pipeline = device.create_compute_pipeline(&cull_signature).unwrap();
        let mut cull_pool = cull_signature.create_pool(0, 1).unwrap();
        let cull_descriptor = cull_pool.alloc().unwrap();

        let cull_buffer = device
            .create_device_buffer(
                BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
                mem::size_of::<CullObject>() as u64,
                MAX_OBJECT_COUNT as u64,
            )
            .unwrap();
        let cull_host_buffer = device
            .create_host_buffer(BufferUsageFlags::TRANSFER_SRC, MAX_OBJECT_COUNT as u64)
            .unwrap();
        let visibility_buffer = device
            .create_device_buffer(
                BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST,
                mem::size_of::<u32>() as u64,
                MAX_OBJECT_COUNT as u64,
            )
            .unwrap();
        let visibility_host_buffer = device
            .create_host_buffer(BufferUsageFlags::TRANSFER_DST, MAX_OBJECT_COUNT as u64)
            .unwrap();

        // Translucency depth pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let translucency_depths_pixel_shader = device
            .create_pixel_shader(include_bytes!(
                "../shaders/build/translucency_closest_depths.frag.spv"
            ))
            .unwrap();

        // Compose pipeline
        // -----------------------------------------------------------------------------------------------------------------
        let quad_vert_shader = device
            .create_vertex_shader(include_bytes!("../shaders/build/quad.vert.spv"), &[])
            .unwrap();
        let compose_pixel_shader = device
            .create_pixel_shader(include_bytes!("../shaders/build/compose.frag.spv"))
            .unwrap();
        let compose_signature = device
            .create_pipeline_signature(&[quad_vert_shader, compose_pixel_shader], &[])
            .unwrap();
        let mut compose_pool = compose_signature.create_pool(0, 1).unwrap();
        let compose_desc = compose_pool.alloc().unwrap();

        // Create G-Buffer pass resources
        // -----------------------------------------------------------------------------------------------------------------
        let g_render_pass = device
            .create_render_pass(
                &[
                    // Albedo
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::InitClearFinalSave,
                    },
                    // Specular
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::FinalSave,
                    },
                    // Emission
                    Attachment {
                        format: Format::RGBA8_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::FinalSave,
                    },
                    // Normal
                    Attachment {
                        format: Format::RG16_UNORM,
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::SHADER_READ,
                        load_store: LoadStore::FinalSave,
                    },
                    // Depth (read)
                    Attachment {
                        format: Format::D32_FLOAT,
                        init_layout: ImageLayout::DEPTH_STENCIL_READ,
                        final_layout: ImageLayout::DEPTH_STENCIL_READ,
                        load_store: LoadStore::InitLoad,
                    },
                ],
                &[
                    // Solid colors pass
                    Subpass {
                        color: vec![
                            AttachmentRef {
                                index: 0,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 1,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 2,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                            AttachmentRef {
                                index: 3,
                                layout: ImageLayout::COLOR_ATTACHMENT,
                            },
                        ],
                        depth: Some(AttachmentRef {
                            index: 4,
                            layout: ImageLayout::DEPTH_STENCIL_ATTACHMENT,
                        }),
                    },
                ],
                &[],
            )
            .unwrap();

        let g_signature = device
            .create_pipeline_signature(&[], &*ADDITIONAL_PIPELINE_BINDINGS)
            .unwrap();
        let mut g_per_frame_pool = g_signature.create_pool(0, 1).unwrap();
        let mut g_dyn_pool = g_signature.create_pool(1, 1).unwrap();
        let g_per_frame_in = g_per_frame_pool.alloc().unwrap();
        let g_dyn_in = g_dyn_pool.alloc().unwrap();

        let depth_secondary_cls = iter::repeat_with(|| {
            graphics_queue
                .create_secondary_cmd_list("depth_secondary")
                .unwrap()
        })
        .take(available_threads)
        .collect();
        let translucency_depths_secondary_cls = iter::repeat_with(|| {
            graphics_queue
                .create_secondary_cmd_list("g_secondary-for-translucency-depths")
                .unwrap()
        })
        .take(available_threads)
        .collect();
        let g_solid_secondary_cls = iter::repeat_with(|| {
            graphics_queue
                .create_secondary_cmd_list("g_secondary-for-solid-objs")
                .unwrap()
        })
        .take(available_threads)
        .collect();
        let g_translucent_secondary_cls = iter::repeat_with(|| {
            graphics_queue
                .create_secondary_cmd_list("g_secondary-for-translucent-objs")
                .unwrap()
        })
        .take(available_threads)
        .collect();

        let tile_count = max_texture_count;
        let texture_atlases = [
            // albedo
            texture_atlas::new(
                device,
                Format::BC7_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
            // specular
            texture_atlas::new(
                device,
                Format::BC7_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
            // emission
            texture_atlas::new(
                device,
                Format::BC7_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
            // normal
            texture_atlas::new(
                device,
                Format::BC5_RG_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
        ];
        let tex_atlas_sampler = device
            .create_sampler(
                SamplerFilter::NEAREST,
                SamplerFilter::LINEAR,
                SamplerMipmap::LINEAR,
                SamplerClamp::REPEAT,
                settings.textures_max_anisotropy,
            )
            .unwrap();

        // Update pipeline inputs
        unsafe {
            device.update_descriptor_set(
                g_per_frame_in,
                &[
                    g_per_frame_pool.create_binding(
                        0,
                        0,
                        BindingRes::Buffer(per_frame_uniform_buffer.handle()),
                    ),
                    g_per_frame_pool.create_binding(1, 0, BindingRes::Buffer(material_buffer.handle())),
                    g_per_frame_pool.create_binding(
                        2,
                        0,
                        BindingRes::Image(
                            Arc::clone(&texture_atlases[0].image()),
                            Some(Arc::clone(&tex_atlas_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                    g_per_frame_pool.create_binding(
                        3,
                        0,
                        BindingRes::Image(
                            Arc::clone(&texture_atlases[1].image()),
                            Some(Arc::clone(&tex_atlas_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                    g_per_frame_pool.create_binding(
                        4,
                        0,
                        BindingRes::Image(
                            Arc::clone(&texture_atlases[3].image()),
                            Some(Arc::clone(&tex_atlas_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                ],
            );
            device.update_descriptor_set(
                g_dyn_in,
                &[g_dyn_pool.create_binding(
                    0,
                    0,
                    BindingRes::BufferRange(uniform_buffer_basic.handle(), 0..MAX_BASIC_UNIFORM_BLOCK_SIZE),
                )],
            );
        }

        let transfer_cl = [
            transfer_queue.create_primary_cmd_list("transfer_0").unwrap(),
            graphics_queue.create_primary_cmd_list("transfer_1").unwrap(),
        ];
        let transfer_submit = [
            device
                .create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&transfer_cl[0])], &[])])
                .unwrap(),
            device
                .create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&transfer_cl[1])], &[])])
                .unwrap(),
        ];

        let staging_cl = graphics_queue.create_primary_cmd_list("staging").unwrap();
        let staging_submit = device
            .create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&staging_cl)], &[])])
            .unwrap();

        let modules_updates_submit = device
            .create_submit_packet(&[SubmitInfo::new(&[], &[], &[])])
            .unwrap();

        let final_cl = [
            graphics_queue.create_primary_cmd_list("final").unwrap(),
            present_queue.create_primary_cmd_list("final").unwrap(),
        ];
        let final_submit = [
            device.create_submit_packet(&[]).unwrap(),
            device.create_submit_packet(&[]).unwrap(),
        ];

        // TODO: allocate necessarry buffers with capacity of MAX_OBJECTS
        let mut renderer = Renderer {
            scene,
            active_camera,
            camera_pos_pivot: Default::default(),
            relative_camera_pos: Default::default(),
            surface: Arc::clone(surface),
            swapchain: None,
            surface_changed: false,
            surface_size: size,
            settings,
            device: Arc::clone(device),
            texture_atlases,
            tex_atlas_sampler,
            staging_buffer,
            transfer_cl,
            transfer_submit,
            staging_cl,
            staging_submit,
            modules_updates_submit,
            final_cl,
            final_submit,
            sw_framebuffers: vec![],
            visibility_buffer,
            depth_secondary_cls,
            translucency_depths_secondary_cls,
            g_solid_secondary_cls,
            g_translucent_secondary_cls,
            depth_render_pass,
            depth_pyramid_image: None,
            depth_pyramid_views: vec![],
            depth_framebuffer: None,
            depth_pyramid_pipeline,
            depth_pyramid_signature,
            depth_pyramid_pool: None,
            depth_pyramid_descs: vec![],
            cull_pipeline,
            cull_signature,
            cull_pool,
            cull_desc: cull_descriptor,
            cull_buffer,
            g_render_pass,
            g_framebuffer: None,
            g_per_frame_pool,
            g_dyn_pool,
            g_per_frame_in,
            g_dyn_in,
            translucency_depths_pixel_shader,
            translucency_depths_image: None,
            translucency_colors_image: None,
            per_frame_ub: per_frame_uniform_buffer,
            visibility_host_buffer,
            sw_render_pass: None,
            compose_pipeline: None,
            compose_signature,
            compose_pool,
            cull_host_buffer,
            renderables: Default::default(),
            material_buffer,
            material_updates: Default::default(),
            compose_desc,
            material_pipelines: vec![],
            uniform_buffer_basic,
            vertex_meshes: Default::default(),
            vertex_mesh_updates: Default::default(),
            vertex_mesh_pending_updates: vec![],
            uniform_buffer_offsets: IndexPool::new(),
            ordered_entities: vec![],
            modules: Default::default(),
        };
        renderer.on_resize(size);

        renderer
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn active_camera(&self) -> &Camera {
        &self.active_camera
    }

    pub fn active_camera_mut(&mut self) -> &mut Camera {
        &mut self.active_camera
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn set_settings(&mut self, settings: Settings) {
        // TODO: change rendering according to settings
        self.settings = settings;
    }

    fn on_update(&mut self) -> UpdateTimings {
        let mut timings = UpdateTimings::default();
        let total_t0 = Instant::now();
        let camera = *self.active_camera();

        self.relative_camera_pos = camera.position() - self.camera_pos_pivot;

        // Reset camera to origin (0, 0, 0) to save rendering precision
        // when camera position is too far (distance > 4096) from origin
        if self.relative_camera_pos.magnitude() >= RESET_CAMERA_POS_THRESHOLD {
            let mut global_transform = self.scene.global_transform();
            global_transform.position -= self.relative_camera_pos;
            self.scene.set_global_transform(global_transform);

            self.relative_camera_pos = DVec3::default();
            self.camera_pos_pivot = camera.position();
        }

        // --------------------------------------------------------------------
        let mut update_cls = vec![];
        for module in self.modules.values_mut() {
            if let Some(cl) = module.on_update(&mut self.scene) {
                update_cls.push(cl);
            }
        }
        self.modules_updates_submit.get_mut().unwrap()[0].set_cmd_lists(update_cls);

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        unsafe { graphics_queue.submit(&mut self.modules_updates_submit).unwrap() };

        // --------------------------------------------------------------------

        let mut uniform_buffer_updates = BufferUpdate2 {
            buffer: self.uniform_buffer_basic.handle(),
            data: smallvec![],
            regions: vec![],
        };
        let mut buffer_updates = vec![];

        // Asynchronously acquire buffers from transfer queue to render queue
        if !self.vertex_mesh_pending_updates.is_empty() {
            let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
            unsafe { graphics_queue.submit(&mut self.transfer_submit[1]).unwrap() };
        }

        let t00 = Instant::now();

        let mut renderer_events_system = system::RendererComponentEvents {
            device: &self.device,
            renderer_comps: self.scene.storage::<component::MeshRenderConfig>(),
            renderables: &mut self.renderables,
            buffer_updates: &mut buffer_updates,
            material_pipelines: &mut self.material_pipelines,
            uniform_buffer_offsets: &mut self.uniform_buffer_offsets,
        };
        let mut vertex_mesh_system = system::VertexMeshCompEvents {
            vertex_meshes: &mut self.vertex_meshes,
            vertex_mesh_comps: self.scene.storage::<component::VertexMesh>(),
            buffer_updates: &mut self.vertex_mesh_updates,
        };
        let mut hierarchy_propagation_system = system::HierarchyPropagation {
            parent_comps: self.scene.storage::<Parent>(),
            children_comps: self.scene.storage::<Children>(),
            transform_comps: self.scene.storage::<component::Transform>(),
            global_transform_comps: self.scene.storage::<GlobalTransform>(),
            ordered_entities: &mut self.ordered_entities,
        };

        rayon::scope(|s| {
            let t0 = Instant::now();
            s.spawn(|_| renderer_events_system.run());
            let t1 = Instant::now();
            s.spawn(|_| vertex_mesh_system.run());
            let t2 = Instant::now();
            s.spawn(|_| hierarchy_propagation_system.run());
            let t3 = Instant::now();

            timings.batch0_render_events = (t1 - t0).as_secs_f64();
            timings.batch0_vertex_meshes = (t2 - t1).as_secs_f64();
            timings.batch0_hierarchy_propag = (t3 - t2).as_secs_f64();

            // Wait for previous transfers before committing them
            if !self.vertex_mesh_pending_updates.is_empty() {
                s.spawn(|_| self.transfer_submit[1].wait().unwrap());
            }
        });

        let t11 = Instant::now();
        timings.systems_batch0 = (t11 - t00).as_secs_f64();

        let mut global_transform_events_system = system::GlobalTransformEvents {
            uniform_buffer_updates: &mut uniform_buffer_updates,
            global_transform_comps: self.scene.storage::<GlobalTransform>(),
            renderer_comps: self.scene.storage::<component::MeshRenderConfig>(),
            material_pipelines: &self.material_pipelines,
            renderables: &self.renderables,
        };

        let t00 = Instant::now();

        // Before updating new buffers, collect all the completed updates to commit them
        let completed_updates = self.vertex_mesh_pending_updates.drain(..).collect();

        // Sort by distance to perform updates of the nearest vertex meshes first
        let mut sorted_buffer_updates_entities: Vec<_> = {
            let transforms = self.scene.storage_read::<GlobalTransform>();
            self.vertex_mesh_updates
                .keys()
                .map(|v| {
                    let transform = transforms.get(*v).unwrap();
                    (
                        *v,
                        (transform.position - self.relative_camera_pos).magnitude_squared(),
                    )
                })
                .collect()
        };
        sorted_buffer_updates_entities.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut buffer_update_system = system::GpuBuffersUpdate {
            device: Arc::clone(&self.device),
            transfer_cl: &self.transfer_cl,
            transfer_submit: &mut self.transfer_submit,
            buffer_updates: &mut self.vertex_mesh_updates,
            sorted_buffer_updates_entities: &sorted_buffer_updates_entities,
            pending_buffer_updates: &mut self.vertex_mesh_pending_updates,
        };
        let commit_buffer_updates_system = system::CommitBufferUpdates {
            updates: completed_updates,
            vertex_meshes: &mut self.vertex_meshes,
            vertex_mesh_comps: self.scene.storage::<component::VertexMesh>(),
        };

        rayon::scope(|s| {
            let t0 = Instant::now();
            s.spawn(|_| global_transform_events_system.run());
            let t1 = Instant::now();
            s.spawn(|_| buffer_update_system.run());
            let t2 = Instant::now();
            s.spawn(|_| commit_buffer_updates_system.run());
            let t3 = Instant::now();

            timings.batch1_global_transforms = (t1 - t0).as_secs_f64();
            timings.batch1_buffer_updates = (t2 - t1).as_secs_f64();
            timings.batch1_updates_commit = (t3 - t2).as_secs_f64();
        });

        // FIXME: VMA: parallel invocations (when creating Cluster meshes) of `vkAllocateMemory` causes huge stutters

        let t11 = Instant::now();
        timings.systems_batch1 = (t11 - t00).as_secs_f64();

        // Update camera uniform buffers
        // -------------------------------------------------------------------------------------------------------------
        {
            let per_frame_info = {
                let cam_pos: Vec3 = glm::convert(self.relative_camera_pos);
                let cam_dir = camera.direction();
                let proj = camera.projection();
                let view = camera::create_view_matrix(glm::convert(cam_pos), camera.rotation());

                FrameInfoUniforms {
                    camera: CameraInfo {
                        pos: Vec4::new(cam_pos.x, cam_pos.y, cam_pos.z, 0.0),
                        dir: Vec4::new(cam_dir.x, cam_dir.y, cam_dir.z, 0.0),
                        proj,
                        view,
                        proj_view: proj * view,
                        z_near: camera.z_near(),
                        fovy: camera.fovy(),
                    },
                    atlas_info: U32Vec4::new(self.texture_atlases[0].tile_width(), 0, 0, 0),
                    frame_size: UVec2::new(self.surface_size.0, self.surface_size.1),
                }
            };

            let data = unsafe {
                slice::from_raw_parts(
                    &per_frame_info as *const FrameInfoUniforms as *const u8,
                    mem::size_of_val(&per_frame_info),
                )
                .to_smallvec()
            };
            buffer_updates.push(BufferUpdate::Type1(BufferUpdate1 {
                buffer: self.per_frame_ub.handle(),
                offset: 0,
                data,
            }));
        }

        // Update material buffer
        // -------------------------------------------------------------------------------------------------------------
        if !self.material_updates.is_empty() {
            let mut data = Vec::<MaterialInfo>::with_capacity(self.material_updates.len());
            let mat_size = mem::size_of::<MaterialInfo>() as u64;

            let regions: Vec<CopyRegion> = self
                .material_updates
                .drain()
                .enumerate()
                .map(|(i, (id, info))| {
                    data.push(info);
                    CopyRegion::new(
                        i as u64 * mat_size,
                        id as u64 * mat_size,
                        (mat_size as u64).try_into().unwrap(),
                    )
                })
                .collect();

            let data =
                unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mat_size as usize) };

            buffer_updates.push(BufferUpdate::Type2(BufferUpdate2 {
                buffer: self.material_buffer.handle(),
                data: data.to_smallvec(),
                regions,
            }));
        }

        buffer_updates.push(BufferUpdate::Type2(uniform_buffer_updates));

        unsafe { self.update_device_buffers(&buffer_updates) };

        let t1 = Instant::now();
        timings.uniform_buffers_update = (t1 - t11).as_secs_f64();

        self.modules_updates_submit.wait().unwrap();

        let total_t1 = Instant::now();
        timings.total = (total_t1 - total_t0).as_secs_f64();

        timings
    }

    fn record_depth_cmd_lists(&mut self) -> u32 {
        let global_transform_comps = self.scene.storage_read::<GlobalTransform>();
        let renderer_comps = self.scene.storage_read::<component::MeshRenderConfig>();

        let mat_pipelines = &self.material_pipelines;
        let object_count = self.ordered_entities.len();
        let draw_count_step = object_count / self.depth_secondary_cls.len() + 1;
        let cull_objects = Mutex::new(Vec::<CullObject>::with_capacity(object_count));

        let proj_mat = self.active_camera.projection();
        let view_mat = camera::create_view_matrix(
            glm::convert(self.relative_camera_pos),
            self.active_camera.rotation(),
        );
        let frustum = Frustum::new(proj_mat * view_mat);

        let translucency_consts = TranslucencyPushConsts {
            frame_size: UVec2::new(self.surface_size.0, self.surface_size.1),
        };

        self.depth_secondary_cls
            .par_iter()
            .zip(&self.translucency_depths_secondary_cls)
            .enumerate()
            .for_each(|(i, (cmd_list_solid, cmd_list_translucency))| {
                let mut curr_cull_objects = Vec::with_capacity(draw_count_step);

                let mut cl_sol = cmd_list_solid.lock();
                let mut cl_trans = cmd_list_translucency.lock();

                cl_sol
                    .begin_secondary_graphics(
                        true,
                        &self.depth_render_pass,
                        0,
                        self.depth_framebuffer.as_ref(),
                    )
                    .unwrap();
                cl_trans
                    .begin_secondary_graphics(
                        true,
                        &self.depth_render_pass,
                        0,
                        self.depth_framebuffer.as_ref(),
                    )
                    .unwrap();

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable_id = self.ordered_entities[entity_index];

                    let global_transform =
                        unwrap_option!(global_transform_comps.get(renderable_id), continue);
                    let render_config = unwrap_option!(renderer_comps.get(renderable_id), continue);
                    let vertex_mesh = unwrap_option!(self.vertex_meshes.get(&renderable_id), continue);

                    if vertex_mesh.vertex_count == 0 && render_config.fake_vertex_count == 0 {
                        continue;
                    }

                    let sphere = vertex_mesh.sphere();
                    let center = sphere.center() + global_transform.position_f32();
                    let radius = sphere.radius() * global_transform.scale.max();

                    if !render_config.visible
                        || (!frustum.is_sphere_visible(&center, radius)
                            && render_config.fake_vertex_count == 0)
                    {
                        continue;
                    }

                    curr_cull_objects.push(CullObject {
                        sphere: Vec4::new(center.x, center.y, center.z, radius),
                        id: entity_index as u32,
                    });

                    let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
                    let renderable = &self.renderables[&renderable_id];

                    let (cl, pipeline_id) = if render_config.translucent {
                        (&mut cl_trans, PIPELINE_TRANSLUCENCY_DEPTHS)
                    } else {
                        (&mut cl_sol, PIPELINE_DEPTH_WRITE)
                    };
                    let pipeline = mat_pipeline.get_pipeline(pipeline_id).unwrap();
                    let signature = pipeline.signature();

                    let already_bound = cl.bind_pipeline(pipeline);
                    if !already_bound {
                        cl.bind_graphics_input(
                            signature,
                            DESC_SET_GENERAL_PER_FRAME,
                            self.g_per_frame_in,
                            &[],
                        );
                        if let Some((_, set)) = &mat_pipeline.custom_per_frame_uniform_desc {
                            cl.bind_graphics_input(signature, DESC_SET_CUSTOM_PER_FRAME, *set, &[]);
                        }
                        if pipeline_id == PIPELINE_TRANSLUCENCY_DEPTHS {
                            cl.push_constants(signature, &translucency_consts);
                        }
                    }

                    cl.bind_graphics_input(
                        pipeline.signature(),
                        DESC_SET_CUSTOM_PER_OBJECT,
                        self.g_dyn_in,
                        &[renderable.uniform_buf_index as u32 * MAX_BASIC_UNIFORM_BLOCK_SIZE as u32],
                    );

                    if vertex_mesh.vertex_count > 0 {
                        cl.bind_and_draw_vertex_mesh(vertex_mesh);
                    } else if vertex_mesh.instance_count > 0 {
                        cl.bind_vertex_buffers(0, &vertex_mesh.bindings());
                        cl.draw_instanced(render_config.fake_vertex_count, 0, 0, vertex_mesh.instance_count);
                    }
                }

                cl_sol.end().unwrap();
                cl_trans.end().unwrap();

                cull_objects.lock().extend(curr_cull_objects);
            });

        let cull_objects = cull_objects.into_inner();
        self.cull_host_buffer.write(0, &cull_objects);

        cull_objects.len() as u32
    }

    fn record_g_cmd_lists(&self) {
        let renderer_comps = self.scene.storage_read::<component::MeshRenderConfig>();

        let mat_pipelines = &self.material_pipelines;
        let object_count = self.ordered_entities.len();
        let draw_count_step = object_count / self.g_solid_secondary_cls.len() + 1;

        self.g_solid_secondary_cls
            .par_iter()
            .zip(&self.g_translucent_secondary_cls)
            .enumerate()
            .for_each(|(i, (cmd_list_solid, cmd_list_translucent))| {
                let mut cl_sol = cmd_list_solid.lock();
                let mut cl_trans = cmd_list_translucent.lock();

                cl_sol
                    .begin_secondary_graphics(true, &self.g_render_pass, 0, self.g_framebuffer.as_ref())
                    .unwrap();
                cl_trans
                    .begin_secondary_graphics(true, &self.g_render_pass, 0, self.g_framebuffer.as_ref())
                    .unwrap();

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable_id = self.ordered_entities[entity_index];
                    let render_config = unwrap_option!(renderer_comps.get(renderable_id), continue);

                    // Occlusion culling can't handle arbitrary vertex representations,
                    // so check for fake_vertex_count also.
                    if self.visibility_host_buffer[entity_index] == 0 && render_config.fake_vertex_count == 0
                    {
                        continue;
                    }

                    let vertex_mesh = unwrap_option!(self.vertex_meshes.get(&renderable_id), continue);
                    let renderable = self.renderables.get(&renderable_id).unwrap();

                    let mut consts = GPassConsts {
                        is_translucent_pass: 0,
                    };

                    let mat_pipeline = &mat_pipelines[render_config.mat_pipeline as usize];
                    let (cl, pipeline_id) = if render_config.translucent {
                        consts.is_translucent_pass = 1;
                        (&mut cl_trans, PIPELINE_COLOR_WITH_BLENDING)
                    } else {
                        (&mut cl_sol, PIPELINE_COLOR)
                    };
                    let pipeline = mat_pipeline.get_pipeline(pipeline_id).unwrap();
                    let signature = pipeline.signature();

                    let already_bound = cl.bind_pipeline(pipeline);
                    if !already_bound {
                        cl.bind_graphics_input(
                            signature,
                            DESC_SET_GENERAL_PER_FRAME,
                            self.g_per_frame_in,
                            &[],
                        );
                        if let Some((_, set)) = &mat_pipeline.custom_per_frame_uniform_desc {
                            cl.bind_graphics_input(signature, DESC_SET_CUSTOM_PER_FRAME, *set, &[]);
                        }
                        cl.push_constants(signature, &consts);
                    }

                    cl.bind_graphics_input(
                        signature,
                        DESC_SET_CUSTOM_PER_OBJECT,
                        self.g_dyn_in,
                        &[renderable.uniform_buf_index as u32 * MAX_BASIC_UNIFORM_BLOCK_SIZE as u32],
                    );

                    if vertex_mesh.vertex_count > 0 {
                        cl.bind_and_draw_vertex_mesh(vertex_mesh);
                    } else {
                        cl.bind_vertex_buffers(0, &vertex_mesh.bindings());
                        cl.draw_instanced(render_config.fake_vertex_count, 0, 0, vertex_mesh.instance_count);
                    }
                }

                cl_sol.end().unwrap();
                cl_trans.end().unwrap();
            });
    }

    fn depth_pass(&mut self) {
        let frustum_visible_objects = self.record_depth_cmd_lists();
        let object_count = self.ordered_entities.len() as u32;

        let mut cl = self.staging_cl.lock();
        cl.begin(true).unwrap();

        let translucency_depths_image = self.translucency_depths_image.as_ref().unwrap();
        cl.fill_buffer(translucency_depths_image, u32::MAX);

        cl.barrier_buffer(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::PIXEL_SHADER,
            &[translucency_depths_image
                .barrier()
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_WRITE | AccessFlags::SHADER_READ)],
        );

        cl.begin_render_pass(
            &self.depth_render_pass,
            self.depth_framebuffer.as_ref().unwrap(),
            &[ClearValue::Depth(1.0)],
            true,
        );

        // Render solid objects
        cl.execute_secondary(&self.depth_secondary_cls);

        // Find closest depths of translucent objects
        cl.execute_secondary(&self.translucency_depths_secondary_cls);

        cl.end_render_pass();

        let depth_image = self.depth_framebuffer.as_ref().unwrap().get_image(0).unwrap();
        let depth_pyramid_image = self.depth_pyramid_image.as_ref().unwrap();

        // Build depth pyramid
        // ----------------------------------------------------------------------------------------

        cl.barrier_image(
            PipelineStageFlags::ALL_GRAPHICS,
            PipelineStageFlags::COMPUTE,
            &[
                depth_image
                    .barrier()
                    .src_access_mask(AccessFlags::DEPTH_ATTACHMENT_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .layout(ImageLayout::SHADER_READ),
                depth_pyramid_image
                    .barrier()
                    .dst_access_mask(AccessFlags::SHADER_WRITE)
                    .old_layout(ImageLayout::UNDEFINED)
                    .new_layout(ImageLayout::GENERAL),
            ],
        );

        cl.bind_pipeline(&self.depth_pyramid_pipeline);

        let mut out_size = depth_pyramid_image.size_2d();

        for i in 0..(depth_pyramid_image.mip_levels() as usize) {
            cl.bind_compute_input(&self.depth_pyramid_signature, 0, self.depth_pyramid_descs[i], &[]);

            let constants = DepthPyramidConstants {
                out_size: Vec2::new(out_size.0 as f32, out_size.1 as f32),
            };
            cl.push_constants(&self.depth_pyramid_signature, &constants);

            cl.dispatch(calc_group_count(out_size.0), calc_group_count(out_size.1), 1);

            cl.barrier_image(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[depth_pyramid_image
                    .barrier()
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .old_layout(ImageLayout::GENERAL)
                    .new_layout(ImageLayout::GENERAL)
                    .mip_levels(i as u32, 1)],
            );

            out_size = ((out_size.0 >> 1).max(1), (out_size.1 >> 1).max(1));
        }

        cl.barrier_image(
            PipelineStageFlags::COMPUTE,
            PipelineStageFlags::BOTTOM_OF_PIPE,
            &[depth_image
                .barrier()
                .src_access_mask(AccessFlags::SHADER_READ)
                .dst_access_mask(Default::default())
                .old_layout(ImageLayout::SHADER_READ)
                .new_layout(ImageLayout::DEPTH_STENCIL_READ)],
        );

        // Compute visibilities
        // ----------------------------------------------------------------------------------------

        cl.copy_buffer_to_device(
            &self.cull_host_buffer,
            0,
            &self.cull_buffer,
            0,
            frustum_visible_objects as u64,
        );
        cl.fill_buffer(&self.visibility_buffer, 0);

        cl.barrier_buffer(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::COMPUTE,
            &[
                self.cull_buffer
                    .barrier()
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ),
                self.visibility_buffer
                    .barrier()
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_WRITE),
            ],
        );

        cl.bind_pipeline(&self.cull_pipeline);
        cl.bind_compute_input(&self.cull_signature, 0, self.cull_desc, &[]);

        let pyramid_size = depth_pyramid_image.size_2d();
        let constants = CullConstants {
            pyramid_size: Vec2::new(pyramid_size.0 as f32, pyramid_size.1 as f32),
            max_pyramid_levels: depth_pyramid_image.mip_levels(),
            object_count: frustum_visible_objects,
        };
        cl.push_constants(&self.cull_signature, &constants);

        cl.dispatch(calc_group_count(frustum_visible_objects), 1, 1);

        cl.barrier_buffer(
            PipelineStageFlags::COMPUTE,
            PipelineStageFlags::TRANSFER,
            &[self
                .visibility_buffer
                .barrier()
                .src_access_mask(AccessFlags::SHADER_WRITE)
                .dst_access_mask(AccessFlags::TRANSFER_READ)],
        );

        cl.copy_buffer_to_host(
            &self.visibility_buffer,
            0,
            &self.visibility_host_buffer,
            0,
            object_count as u64,
        );

        cl.end().unwrap();
        drop(cl);

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        let submit = &mut self.staging_submit;
        unsafe { graphics_queue.submit(submit).unwrap() };
        submit.wait().unwrap();
    }

    fn on_render(&mut self, sw_image: &SwapchainImage) -> RenderTimings {
        let mut timings = RenderTimings::default();
        let total_t0 = Instant::now();
        let device = Arc::clone(&self.device);
        let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);

        let t1 = Instant::now();

        self.depth_pass();

        let t2 = Instant::now();
        timings.depth_pass = (t2 - t1).as_secs_f64();

        self.record_g_cmd_lists();

        let t3 = Instant::now();
        // FIXME
        // timings.color_ex = (t3 - t2).as_secs_f64();

        let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);

        // Record G-Buffer cmd list
        // -------------------------------------------------------------------------------------------------------------
        let mut cl = self.final_cl[0].lock();
        cl.begin(true).unwrap();

        let translucency_colors_image = self.translucency_colors_image.as_ref().unwrap();

        cl.barrier_image(
            PipelineStageFlags::TOP_OF_PIPE,
            PipelineStageFlags::TRANSFER,
            &[translucency_colors_image
                .barrier()
                .dst_access_mask(AccessFlags::TRANSFER_WRITE)
                .old_layout(ImageLayout::UNDEFINED)
                .new_layout(ImageLayout::GENERAL)],
        );
        cl.clear_image(
            translucency_colors_image,
            ImageLayout::GENERAL,
            ClearValue::ColorF32([0.0; 4]),
        );
        cl.barrier_image(
            PipelineStageFlags::TRANSFER,
            PipelineStageFlags::PIXEL_SHADER,
            &[translucency_colors_image
                .barrier()
                .src_access_mask(AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_WRITE)
                .layout(ImageLayout::GENERAL)],
        );

        // Render solid objects
        cl.begin_render_pass(
            &self.g_render_pass,
            self.g_framebuffer.as_ref().unwrap(),
            &[
                ClearValue::ColorF32([0.0, 0.0, 0.0, 1.0]),
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::Undefined,
                ClearValue::ColorU32([0xffffffff; 4]),
            ],
            true,
        );
        cl.execute_secondary(&self.g_solid_secondary_cls);
        cl.execute_secondary(&self.g_translucent_secondary_cls);
        cl.end_render_pass();

        let albedo = self.g_framebuffer.as_ref().unwrap().get_image(0).unwrap();

        cl.barrier_image(
            PipelineStageFlags::ALL_GRAPHICS,
            PipelineStageFlags::PIXEL_SHADER,
            &[
                albedo
                    .barrier()
                    .src_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .layout(ImageLayout::SHADER_READ),
                translucency_colors_image
                    .barrier()
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)
                    .layout(ImageLayout::GENERAL),
            ],
        );

        // Compose final swapchain image
        cl.begin_render_pass(
            self.sw_render_pass.as_ref().unwrap(),
            &self.sw_framebuffers[sw_image.index() as usize],
            &[],
            false,
        );
        cl.bind_pipeline(self.compose_pipeline.as_ref().unwrap());
        cl.bind_graphics_input(&self.compose_signature, 0, self.compose_desc, &[]);
        cl.draw(3, 0);
        cl.end_render_pass();

        if graphics_queue != present_queue {
            cl.barrier_image(
                PipelineStageFlags::ALL_GRAPHICS,
                PipelineStageFlags::BOTTOM_OF_PIPE,
                &[sw_image
                    .get()
                    .barrier()
                    .src_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .old_layout(ImageLayout::PRESENT)
                    .new_layout(ImageLayout::PRESENT)
                    .src_queue(graphics_queue)
                    .dst_queue(present_queue)],
            );
        }

        cl.end().unwrap();
        drop(cl);

        unsafe {
            graphics_queue.submit(&mut self.final_submit[0]).unwrap();

            if graphics_queue != present_queue {
                {
                    let mut cl = self.final_cl[1].lock();
                    cl.begin(true).unwrap();
                    cl.barrier_image(
                        PipelineStageFlags::TOP_OF_PIPE,
                        PipelineStageFlags::BOTTOM_OF_PIPE,
                        &[sw_image
                            .get()
                            .barrier()
                            .old_layout(ImageLayout::PRESENT)
                            .new_layout(ImageLayout::PRESENT)
                            .src_queue(graphics_queue)
                            .dst_queue(present_queue)],
                    );
                    cl.end().unwrap();
                }

                self.final_submit[1]
                    .set(&[SubmitInfo::new(
                        &[WaitSemaphore {
                            semaphore: Arc::clone(graphics_queue.timeline_semaphore()),
                            wait_dst_mask: PipelineStageFlags::ALL_COMMANDS,
                            wait_value: self.final_submit[0].get_signal_value(0).unwrap(),
                        }],
                        &[Arc::clone(&self.final_cl[1])],
                        &[SignalSemaphore {
                            semaphore: Arc::clone(present_queue.end_of_frame_semaphore()),
                            signal_value: 0,
                        }],
                    )])
                    .unwrap();

                present_queue.submit(&mut self.final_submit[1]).unwrap();
            }
        }

        let t3 = Instant::now();
        timings.color_pass = (t3 - t2).as_secs_f64();

        let total_t1 = Instant::now();
        timings.total = (total_t1 - total_t0).as_secs_f64();

        timings
    }

    pub fn on_draw(&mut self) -> RendererTimings {
        let mut timings = RendererTimings::default();
        let device = Arc::clone(&self.device);
        let adapter = device.adapter();
        let surface = &self.surface;

        if !adapter.is_surface_valid(surface).unwrap() {
            return timings;
        }

        // Wait for previous frame completion
        self.final_submit[0].wait().unwrap();
        self.final_submit[1].wait().unwrap();

        if self.surface_changed {
            let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
            let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);

            self.sw_framebuffers.clear();

            self.swapchain = Some(
                device
                    .create_swapchain(
                        &self.surface,
                        self.surface_size,
                        self.settings.fps_limit == FPSLimit::VSync,
                        if self.settings.prefer_triple_buffering {
                            3
                        } else {
                            2
                        },
                        self.swapchain.take(),
                    )
                    .unwrap(),
            );

            let signal_sem = &[SignalSemaphore {
                semaphore: Arc::clone(present_queue.end_of_frame_semaphore()),
                signal_value: 0,
            }];

            self.final_submit[0]
                .set(&[SubmitInfo::new(
                    &[WaitSemaphore {
                        semaphore: Arc::clone(self.swapchain.as_ref().unwrap().readiness_semaphore()),
                        wait_dst_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                        wait_value: 0,
                    }],
                    &[Arc::clone(&self.final_cl[0])],
                    if graphics_queue == present_queue {
                        signal_sem
                    } else {
                        &[]
                    },
                )])
                .unwrap();

            self.create_main_framebuffers();
            self.surface_changed = false;
        }

        let acquire_result = self.swapchain.as_ref().unwrap().acquire_image();

        match acquire_result {
            Ok((sw_image, suboptimal)) => {
                self.surface_changed |= suboptimal;

                timings.update = self.on_update();
                timings.render = self.on_render(&sw_image);

                let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);
                let present_result = present_queue.present(sw_image);

                match present_result {
                    Ok(suboptimal) => {
                        self.surface_changed |= suboptimal;
                    }
                    Err(swapchain::Error::IncompatibleSurface) => {
                        self.surface_changed = true;
                    }
                    _ => {
                        present_result.unwrap();
                    }
                }
            }
            Err(swapchain::Error::IncompatibleSurface) => {
                self.surface_changed = true;
            }
            _ => {
                acquire_result.unwrap();
            }
        }

        timings
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_size = new_size;
        self.surface_changed = true;

        self.device.wait_idle().unwrap();

        // Set camera aspect
        self.active_camera.set_aspect(new_size.0, new_size.1);

        let depth_image = self
            .device
            .create_image_2d(
                Format::D32_FLOAT,
                1,
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
                new_size,
            )
            .unwrap();
        self.depth_pyramid_image = Some(
            self.device
                .create_image_2d_named(
                    Format::R32_FLOAT,
                    0,
                    ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
                    // Note: prev_power_of_two makes sure all reductions are at most by 2x2
                    // which makes sure they are conservative
                    (
                        utils::prev_power_of_two(new_size.0),
                        utils::prev_power_of_two(new_size.1),
                    ),
                    "depth_pyramid",
                )
                .unwrap(),
        );
        let depth_pyramid_image = self.depth_pyramid_image.as_ref().unwrap();
        let depth_pyramid_levels = depth_pyramid_image.mip_levels();
        self.depth_pyramid_views = (0..depth_pyramid_levels)
            .map(|i| {
                depth_pyramid_image
                    .create_view_named(&format!("view-mip{}", i))
                    .base_mip_level(i)
                    .mip_level_count(1)
                    .build()
                    .unwrap()
            })
            .collect();

        let mut depth_pyramid_pool = self
            .depth_pyramid_signature
            .create_pool(0, depth_pyramid_levels)
            .unwrap();
        self.depth_pyramid_descs = {
            let pool = &mut depth_pyramid_pool;

            (0..depth_pyramid_levels as usize)
                .map(|i| {
                    let set = pool.alloc().unwrap();
                    unsafe {
                        self.device.update_descriptor_set(
                            set,
                            &[
                                pool.create_binding(
                                    0,
                                    0,
                                    if i == 0 {
                                        BindingRes::ImageView(
                                            Arc::clone(depth_image.view()),
                                            None,
                                            ImageLayout::SHADER_READ,
                                        )
                                    } else {
                                        BindingRes::ImageView(
                                            Arc::clone(&self.depth_pyramid_views[i - 1]),
                                            None,
                                            ImageLayout::GENERAL,
                                        )
                                    },
                                ),
                                pool.create_binding(
                                    1,
                                    0,
                                    BindingRes::ImageView(
                                        Arc::clone(&self.depth_pyramid_views[i]),
                                        None,
                                        ImageLayout::GENERAL,
                                    ),
                                ),
                            ],
                        )
                    };
                    set
                })
                .collect()
        };
        self.depth_pyramid_pool = Some(depth_pyramid_pool);

        unsafe {
            self.device.update_descriptor_set(
                self.cull_desc,
                &[
                    self.cull_pool.create_binding(
                        0,
                        0,
                        BindingRes::Image(Arc::clone(depth_pyramid_image), None, ImageLayout::GENERAL),
                    ),
                    self.cull_pool
                        .create_binding(1, 0, BindingRes::Buffer(self.per_frame_ub.handle())),
                    self.cull_pool
                        .create_binding(2, 0, BindingRes::Buffer(self.cull_buffer.handle())),
                    self.cull_pool
                        .create_binding(3, 0, BindingRes::Buffer(self.visibility_buffer.handle())),
                ],
            )
        };

        self.depth_framebuffer = Some(
            self.depth_render_pass
                .create_framebuffer(
                    new_size,
                    &[(0, ImageMod::OverrideImage(Arc::clone(&depth_image)))],
                )
                .unwrap(),
        );

        self.g_framebuffer = Some(
            self.g_render_pass
                .create_framebuffer(
                    new_size,
                    &[
                        (
                            0,
                            ImageMod::AdditionalUsage(
                                ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::SAMPLED,
                            ),
                        ),
                        (1, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                        (2, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                        (3, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                        (4, ImageMod::OverrideImage(Arc::clone(&depth_image))),
                        (
                            5,
                            ImageMod::OverrideImage(
                                self.device
                                    .create_image_2d(Format::R32_UINT, 1, ImageUsageFlags::STORAGE, new_size)
                                    .unwrap(),
                            ),
                        ),
                    ],
                )
                .unwrap(),
        );

        // Note: use buffer instead of image because Metal (on macs) does not support atomic ops on images
        self.translucency_depths_image = Some(
            self.device
                .create_device_buffer_named(
                    BufferUsageFlags::STORAGE | BufferUsageFlags::TRANSFER_DST,
                    mem::size_of::<u32>() as u64,
                    (new_size.0 * new_size.1 * TRANSLUCENCY_N_DEPTH_LAYERS) as u64,
                    "translucency_depths",
                )
                .unwrap(),
        );
        self.translucency_colors_image = Some(
            self.device
                .create_image_2d_array_named(
                    Format::RGBA8_UNORM,
                    1,
                    ImageUsageFlags::STORAGE | ImageUsageFlags::TRANSFER_DST,
                    (new_size.0, new_size.1, TRANSLUCENCY_N_DEPTH_LAYERS),
                    "translucency_colors",
                )
                .unwrap(),
        );

        unsafe {
            self.device.update_descriptor_set(
                self.g_per_frame_in,
                &[
                    self.g_per_frame_pool.create_binding(
                        5,
                        0,
                        BindingRes::Buffer(self.translucency_depths_image.as_ref().unwrap().handle()),
                    ),
                    self.g_per_frame_pool.create_binding(
                        6,
                        0,
                        BindingRes::Image(
                            Arc::clone(self.translucency_colors_image.as_ref().unwrap()),
                            None,
                            ImageLayout::GENERAL,
                        ),
                    ),
                ],
            );

            let albedo = self.g_framebuffer.as_ref().unwrap().get_image(0).unwrap();
            unsafe {
                self.device.update_descriptor_set(
                    self.compose_desc,
                    &[
                        self.compose_pool.create_binding(
                            0,
                            0,
                            BindingRes::Image(Arc::clone(albedo), None, ImageLayout::SHADER_READ),
                        ),
                        self.compose_pool.create_binding(
                            1,
                            0,
                            BindingRes::Buffer(self.per_frame_ub.handle()),
                        ),
                        self.compose_pool.create_binding(
                            5,
                            0,
                            BindingRes::Buffer(self.translucency_depths_image.as_ref().unwrap().handle()),
                        ),
                        self.compose_pool.create_binding(
                            6,
                            0,
                            BindingRes::Image(
                                Arc::clone(self.translucency_colors_image.as_ref().unwrap()),
                                None,
                                ImageLayout::GENERAL,
                            ),
                        ),
                    ],
                )
            };
        }
    }

    fn create_main_framebuffers(&mut self) {
        let images = self.swapchain.as_ref().unwrap().images();

        self.sw_render_pass = Some(
            self.device
                .create_render_pass(
                    &[Attachment {
                        format: images[0].format(),
                        init_layout: ImageLayout::UNDEFINED,
                        final_layout: ImageLayout::PRESENT,
                        load_store: LoadStore::FinalSave,
                    }],
                    &[Subpass {
                        color: vec![AttachmentRef {
                            index: 0,
                            layout: ImageLayout::COLOR_ATTACHMENT,
                        }],
                        depth: None,
                    }],
                    &[],
                )
                .unwrap(),
        );

        self.sw_framebuffers.clear();
        for img in images {
            self.sw_framebuffers.push(
                self.sw_render_pass
                    .as_ref()
                    .unwrap()
                    .create_framebuffer(
                        images[0].size_2d(),
                        &[(0, ImageMod::OverrideImage(Arc::clone(img)))],
                    )
                    .unwrap(),
            );
        }

        self.compose_pipeline = Some(
            self.device
                .create_graphics_pipeline(
                    self.sw_render_pass.as_ref().unwrap(),
                    0,
                    PrimitiveTopology::TRIANGLE_LIST,
                    Default::default(),
                    Default::default(),
                    &[],
                    &self.compose_signature,
                )
                .unwrap(),
        );
    }

    pub fn register_module<M: RendererModule + Send + Sync + 'static>(&mut self, module: M) {
        self.modules.insert(TypeId::of::<M>(), Box::new(module));
    }

    pub fn module<M: RendererModule + Send + Sync + 'static>(&self) -> Option<&M> {
        self.modules
            .get(&TypeId::of::<M>())
            .map(|m| m.as_any().downcast_ref::<M>().unwrap())
    }

    pub fn module_mut<M: RendererModule + Send + Sync + 'static>(&mut self) -> Option<&mut M> {
        self.modules
            .get_mut(&TypeId::of::<M>())
            .map(|m| m.as_any_mut().downcast_mut::<M>().unwrap())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        // Safe pipeline cache
        let pl_cache = self.device.get_pipeline_cache().unwrap();
        fs::write(*PIPELINE_CACHE_FILENAME, pl_cache).unwrap();

        self.device.wait_idle().unwrap();
    }
}
