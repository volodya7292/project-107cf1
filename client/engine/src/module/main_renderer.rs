//! ## Overall rendering pipeline
//! 1. Depth pass  
//! 1.1. Render depth image of solid objects.  
//! 1.2. Construct depth pyramid.  
//! 1.3. Use depth pyramid to cull objects by occlusion.  
//! 1.4. Render translucent objects: find `TRANSLUCENCY_DEPTH_LAYERS` closest depths.
//! 4. Render solid objects into g-buffer (with early-z testing).  
//! 5. Render translucent objects: match depths with respective colors.
//! 6. Compose solid and translucent colors.

mod texture_atlas;

#[macro_use]
pub mod material_pipeline;
#[macro_use]
pub mod vertex_mesh;
pub mod camera;
pub mod component;
pub(crate) mod gpu_executor;
pub mod material;
mod resource_manager;
pub(crate) mod resources;
mod stage;
mod stage_manager;

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

use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::render_config::RenderType;
use crate::ecs::component::uniform_data::BASIC_UNIFORM_BLOCK_MAX_SIZE;
use crate::ecs::component::{MeshRenderConfigC, TransformC, UniformDataC, VertexMeshC};
use crate::ecs::system;
use crate::event::WSIEvent;
use crate::module::main_renderer::camera::OrthoCamera;
use crate::module::main_renderer::gpu_executor::{GPUJob, GPUJobDeviceExt, GPUJobExecInfo};
use crate::module::main_renderer::material::MatComponent;
use crate::module::main_renderer::material_pipeline::MaterialPipelineSet;
use crate::module::main_renderer::resources::{MaterialPipelineParams, RendererResources};
use crate::module::main_renderer::stage::compose::ComposeStage;
use crate::module::main_renderer::stage::depth::DepthStage;
use crate::module::main_renderer::stage::g_buffer::GBufferStage;
use crate::module::main_renderer::stage::present_queue_transition::PresentQueueTransitionStage;
use crate::module::main_renderer::stage::{RenderStage, StageContext};
use crate::module::main_renderer::stage_manager::StageManager;
use crate::module::scene::change_manager::ComponentChangesHandle;
use crate::module::scene::SceneObject;
use crate::module::scene::{Scene, N_MAX_OBJECTS};
use crate::module::EngineModule;
use crate::utils::wsi::{real_window_size, WSISize};
use crate::EngineContext;
use basis_universal::{TranscodeParameters, TranscoderTextureFormat};
use camera::PerspectiveCamera;
use common::glm;
use common::glm::{DVec3, Mat4, U32Vec2, U32Vec4, UVec2, Vec3, Vec4};
use common::lrc::Lrc;
use common::lrc::LrcExt;
use common::nalgebra::{Matrix4, Vector4};
use common::resource_file::ResourceRef;
use common::scene::relation::Relation;
use common::types::HashMap;
use entity_data::{Archetype, EntityId, System, SystemHandler};
use index_pool::IndexPool;
use lazy_static::lazy_static;
use smallvec::{smallvec, SmallVec, ToSmallVec};
use std::fmt::{Display, Formatter};
use std::sync::Arc;
use std::time::{Duration, Instant};
use std::{fs, mem, slice};
use vertex_mesh::RawVertexMesh;
pub use vertex_mesh::VertexMesh;
use vk_wrapper as vkw;
use vk_wrapper::buffer::{BufferHandle, BufferHandleImpl};
use vk_wrapper::sampler::SamplerClamp;
use vk_wrapper::{
    swapchain, BindingLoc, BindingRes, BindingType, BufferUsageFlags, CopyRegion, DescriptorSet, Device,
    Format, HostBuffer, ImageLayout, PrimitiveTopology, QueueType, SamplerFilter, SamplerMipmap, Semaphore,
    Shader, ShaderBinding, ShaderStageFlags, Surface, Swapchain, SwapchainImage,
};
use winit::window::Window;

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

pub type UpdateHandler = dyn Fn(&EngineContext) -> Option<Lrc<GPUJob>>;

pub struct MainRenderer {
    root_entity: EntityId,
    render_config_component_changes: ComponentChangesHandle,
    mesh_component_changes: ComponentChangesHandle,
    transform_component_changes: ComponentChangesHandle,
    uniform_data_component_changes: ComponentChangesHandle,

    active_camera: PerspectiveCamera,
    overlay_camera: OrthoCamera,
    camera_pos_pivot: DVec3,
    relative_camera_pos: Vec3,

    surface: Arc<Surface>,
    swapchain: Option<Arc<Swapchain>>,
    frame_completion_semaphore: Arc<Semaphore>,
    surface_changed: bool,
    surface_size: U32Vec2,
    render_size: U32Vec2,
    settings: Settings,
    last_frame_ts: Instant,
    device: Arc<Device>,

    staging_buffer: HostBuffer<u8>,
    transfer_jobs: ParallelJob,
    staging_job: GPUJob,

    material_updates: HashMap<u32, MaterialInfo>,
    new_vertex_mesh_updates: HashMap<EntityId, Arc<RawVertexMesh>>,
    vertex_mesh_pending_updates: HashMap<EntityId, Arc<RawVertexMesh>>,

    /// Entities ordered in respect to children order inside `Children` components:
    /// global parents are not in order, but all the children are.
    /// Parents are ordered first.
    ordered_entities: Vec<EntityId>,

    res: RendererResources,
    update_handlers: Vec<Box<UpdateHandler>>,
    stage_manager: StageManager,
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

#[derive(Debug, Default)]
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

#[derive(Debug, Default)]
#[repr(C)]
struct FrameInfoUniforms {
    camera: CameraInfo,
    atlas_info: U32Vec4,
    frame_size: UVec2,
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

impl MaterialInfo {
    pub fn new(
        diffuse: MatComponent,
        specular: MatComponent,
        normal_tex_id: u16,
        emission: Vec4,
    ) -> MaterialInfo {
        let mut info = MaterialInfo {
            diffuse_tex_id: u32::MAX,
            specular_tex_id: u32::MAX,
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

pub(crate) struct BufferUpdate1 {
    pub buffer: BufferHandle,
    pub dst_offset: u64,
    pub data: SmallVec<[u8; 256]>,
}

pub(crate) struct BufferUpdate2 {
    pub buffer: BufferHandle,
    pub data: SmallVec<[u8; 256]>,
    pub regions: Vec<CopyRegion>,
}

pub(crate) enum BufferUpdate {
    WithOffset(BufferUpdate1),
    Regions(BufferUpdate2),
}

pub(crate) struct ParallelJob {
    pub work: GPUJob,
    pub sync: GPUJob,
}

pub const TEXTURE_ID_NONE: u16 = u16::MAX;

pub const N_MAX_MATERIALS: u32 = 4096;
pub const COMPUTE_LOCAL_THREADS: u32 = 32;

const RESET_CAMERA_POS_THRESHOLD: f64 = 4096.0;

#[repr(u32)]
pub enum PipelineKind {
    DepthWrite,
    TranslucencyDepths,
    Color,
    ColorWithBlending,
    Overlay,
}

lazy_static! {
    static ref PIPELINE_CACHE_FILENAME: &'static str = if cfg!(debug_assertions) {
        "pipeline_cache-debug"
    } else {
        "pipeline_cache"
    };

    static ref ADDITIONAL_PIPELINE_BINDINGS: [(BindingLoc, ShaderBinding); 9] = [
        // Per frame info
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_FRAME_INFO),
            ShaderBinding {
                stage_flags: ShaderStageFlags::VERTEX | ShaderStageFlags::PIXEL,
                binding_type: BindingType::UNIFORM_BUFFER_DYNAMIC,
                count: 1,
            },
        ),
        // Material buffer
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_MATERIAL_BUFFER),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::STORAGE_BUFFER,
                count: 1,
            },
        ),
        // Albedo atlas
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_ALBEDO_ATLAS),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::SAMPLED_IMAGE,
                count: 1,
            },
        ),
        // Specular atlas
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_SPECULAR_ATLAS),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::SAMPLED_IMAGE,
                count: 1,
            },
        ),
        // Normal atlas
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_NORMAL_ATLAS),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::SAMPLED_IMAGE,
                count: 1,
            },
        ),
        // Translucency depths (only used in translucency passes)
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_TRANSPARENCY_DEPTHS),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::STORAGE_BUFFER,
                count: 1,
            },
        ),
        // Translucency colors (only used in translucency passes)
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_TRANSPARENCY_COLORS),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::STORAGE_IMAGE,
                count: 1,
            },
        ),
        // Solid depths attachment (only used in translucency depths pass)
        (
            BindingLoc::new(shader_ids::SET_GENERAL_PER_FRAME, shader_ids::BINDING_SOLID_DEPTHS),
            ShaderBinding {
                stage_flags: ShaderStageFlags::PIXEL,
                binding_type: BindingType::INPUT_ATTACHMENT,
                count: 1,
            },
        ),
        // Per object info
        (
            BindingLoc::new(shader_ids::SET_PER_OBJECT, shader_ids::BINDING_OBJECT_INFO),
            ShaderBinding {
                stage_flags: ShaderStageFlags::VERTEX,
                binding_type: BindingType::UNIFORM_BUFFER_DYNAMIC,
                count: 1,
            },
        ),
    ];
}

fn calc_group_count(thread_count: u32) -> u32 {
    (thread_count + COMPUTE_LOCAL_THREADS - 1) / COMPUTE_LOCAL_THREADS
}

fn compose_descriptor_sets(
    per_frame_general: DescriptorSet,
    per_frame_custom: DescriptorSet,
    per_object_general: DescriptorSet,
) -> [DescriptorSet; 3] {
    let mut v = [DescriptorSet::default(); 3];
    v[shader_ids::SET_GENERAL_PER_FRAME as usize] = per_frame_general;
    v[shader_ids::SET_CUSTOM_PER_FRAME as usize] = per_frame_custom;
    v[shader_ids::SET_PER_OBJECT as usize] = per_object_general;
    v
}

#[derive(Archetype)]
pub struct VertexMeshObject {
    global_transform: GlobalTransformC,
    relation: Relation,
    transform: TransformC,
    uniforms: UniformDataC,
    render_config: MeshRenderConfigC,
    mesh: VertexMeshC,
}

impl VertexMeshObject {
    pub fn new(transform: TransformC, render_config: MeshRenderConfigC, mesh: VertexMeshC) -> Self {
        Self {
            global_transform: Default::default(),
            relation: Default::default(),
            transform,
            uniforms: Default::default(),
            render_config,
            mesh,
        }
    }
}

impl SceneObject for VertexMeshObject {}

#[derive(Archetype)]
pub struct SimpleObject {
    global_transform: GlobalTransformC,
    relation: Relation,
    transform: TransformC,
}

impl SimpleObject {
    pub fn new() -> Self {
        Self {
            global_transform: Default::default(),
            relation: Default::default(),
            transform: Default::default(),
        }
    }
}

impl SceneObject for SimpleObject {}

impl MainRenderer {
    pub fn new<F: Fn(&[Arc<vkw::Adapter>]) -> usize>(
        name: &str,
        window: &Window,
        settings: Settings,
        max_texture_count: u32,
        adapter_selector: F,
        root_entity: EntityId,
        ctx: &EngineContext,
    ) -> MainRenderer {
        let vke = vkw::Entry::new().unwrap();
        let instance = vke.create_instance(name, window).unwrap();
        let surface = instance.create_surface(window).unwrap();
        let adapters = instance.enumerate_adapters(Some(&surface)).unwrap();

        let adapter_idx = adapter_selector(&adapters);
        let adapter = &adapters[adapter_idx];
        let device = adapter.create_device().unwrap();

        // ------------------------------------------------------------------------------------------------

        let scene = ctx.module_mut::<Scene>();
        let mut change_manager = scene.change_manager_mut();
        let render_config_component_changes = change_manager.register_component_flow::<MeshRenderConfigC>();
        let mesh_component_changes = change_manager.register_component_flow::<VertexMeshC>();
        let transform_component_changes = change_manager.register_component_flow::<TransformC>();
        let uniform_data_component_changes = change_manager.register_component_flow::<UniformDataC>();

        // ------------------------------------------------------------------------------------------------

        let curr_size = real_window_size(window);

        // Load pipeline cache
        if let Ok(res) = fs::read(*PIPELINE_CACHE_FILENAME) {
            device.load_pipeline_cache(&res).unwrap();
        }

        let active_camera = PerspectiveCamera::new(1.0, std::f32::consts::FRAC_PI_2, 0.1);
        let overlay_camera = OrthoCamera::new();

        let staging_buffer = device
            .create_host_buffer(BufferUsageFlags::TRANSFER_SRC, 0x800000)
            .unwrap();
        let per_frame_ub = device
            .create_device_buffer(
                BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM,
                mem::size_of::<FrameInfoUniforms>() as u64,
                2, // For two different RenderStages
            )
            .unwrap();
        let uniform_buffer_basic = device
            .create_device_buffer(
                BufferUsageFlags::UNIFORM | BufferUsageFlags::TRANSFER_DST,
                BASIC_UNIFORM_BLOCK_MAX_SIZE as u64,
                N_MAX_OBJECTS as u64,
            )
            .unwrap();
        // TODO: allow different alignments
        assert_eq!(
            uniform_buffer_basic.aligned_element_size(),
            BASIC_UNIFORM_BLOCK_MAX_SIZE as u64
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
                N_MAX_MATERIALS as u64,
            )
            .unwrap();

        let g_signature = device
            .create_pipeline_signature(&[], &*ADDITIONAL_PIPELINE_BINDINGS)
            .unwrap();
        let mut g_per_frame_pool = g_signature
            .create_pool(shader_ids::SET_GENERAL_PER_FRAME, 1)
            .unwrap();
        let g_per_frame_in = g_per_frame_pool.alloc().unwrap();

        let tile_count = max_texture_count;
        let texture_atlases = [
            // albedo
            texture_atlas::new(
                &device,
                Format::BC7_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
            // specular
            texture_atlas::new(
                &device,
                Format::BC7_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
            // emission
            texture_atlas::new(
                &device,
                Format::BC7_UNORM,
                settings.textures_mipmaps,
                tile_count,
                settings.texture_quality as u32,
            )
            .unwrap(),
            // normal
            texture_atlas::new(
                &device,
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
                        shader_ids::BINDING_FRAME_INFO,
                        0,
                        BindingRes::BufferRange(
                            per_frame_ub.handle(),
                            0..mem::size_of::<FrameInfoUniforms>() as u64,
                        ),
                    ),
                    g_per_frame_pool.create_binding(
                        shader_ids::BINDING_MATERIAL_BUFFER,
                        0,
                        BindingRes::Buffer(material_buffer.handle()),
                    ),
                    g_per_frame_pool.create_binding(
                        shader_ids::BINDING_ALBEDO_ATLAS,
                        0,
                        BindingRes::Image(
                            Arc::clone(&texture_atlases[0].image()),
                            Some(Arc::clone(&tex_atlas_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                    g_per_frame_pool.create_binding(
                        shader_ids::BINDING_SPECULAR_ATLAS,
                        0,
                        BindingRes::Image(
                            Arc::clone(&texture_atlases[1].image()),
                            Some(Arc::clone(&tex_atlas_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                    g_per_frame_pool.create_binding(
                        shader_ids::BINDING_NORMAL_ATLAS,
                        0,
                        BindingRes::Image(
                            Arc::clone(&texture_atlases[3].image()),
                            Some(Arc::clone(&tex_atlas_sampler)),
                            ImageLayout::SHADER_READ,
                        ),
                    ),
                ],
            );
        }

        let transfer_jobs = ParallelJob {
            work: device.create_job("transfer", QueueType::Transfer).unwrap(),
            sync: device.create_job("transfer-sync", QueueType::Graphics).unwrap(),
        };

        let staging_job = device
            .create_job("renderer-staging", QueueType::Graphics)
            .unwrap();

        let frame_completion_semaphore = Arc::new(device.create_binary_semaphore().unwrap());

        let resources = RendererResources {
            texture_atlases,
            _tex_atlas_sampler: tex_atlas_sampler,
            g_per_frame_pool,
            g_per_frame_in,
            per_frame_ub,
            material_buffer,
            uniform_buffer_basic,
            uniform_buffer_offsets: IndexPool::new(),
            renderables: HashMap::with_capacity(N_MAX_OBJECTS as usize),
            material_pipelines: vec![],
            curr_vertex_meshes: HashMap::with_capacity(N_MAX_OBJECTS as usize),
        };

        let graphics_queue = device.get_queue(QueueType::Graphics);
        let present_queue = device.get_queue(QueueType::Present);

        let mut stages: Vec<Box<dyn RenderStage>> = vec![
            Box::new(DepthStage::new(&device)),
            Box::new(GBufferStage::new(&device)),
            Box::new(ComposeStage::new(&device)),
        ];
        if graphics_queue != present_queue {
            stages.push(Box::new(PresentQueueTransitionStage::new(&device)))
        }

        let mut mat_pipe_counter = 0;
        for stage in &mut stages {
            let num_pipes = stage.num_pipeline_kinds();
            stage.setup(&(mat_pipe_counter..mat_pipe_counter + num_pipes).collect::<Vec<_>>());
            mat_pipe_counter += num_pipes;
        }

        let stage_manager = StageManager::new(&device, stages);

        let mut renderer = MainRenderer {
            root_entity,
            render_config_component_changes,
            mesh_component_changes,
            transform_component_changes,
            uniform_data_component_changes,
            active_camera,
            overlay_camera,
            camera_pos_pivot: Default::default(),
            relative_camera_pos: Default::default(),
            surface,
            swapchain: None,
            frame_completion_semaphore,
            surface_changed: false,
            surface_size: glm::convert_unchecked(curr_size.real()),
            render_size: glm::convert_unchecked(curr_size.real()),
            settings,
            last_frame_ts: Instant::now(),
            device,
            staging_buffer,
            transfer_jobs,
            staging_job,
            material_updates: Default::default(),
            new_vertex_mesh_updates: HashMap::with_capacity(1024),
            vertex_mesh_pending_updates: HashMap::with_capacity(1024),
            ordered_entities: Vec::with_capacity(N_MAX_OBJECTS as usize),
            res: resources,
            update_handlers: vec![],
            stage_manager,
        };
        renderer.on_resize(curr_size);

        renderer
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn active_camera(&self) -> &PerspectiveCamera {
        &self.active_camera
    }

    pub fn active_camera_mut(&mut self) -> &mut PerspectiveCamera {
        &mut self.active_camera
    }

    pub fn settings(&self) -> &Settings {
        &self.settings
    }

    pub fn set_settings(&mut self, settings: Settings) {
        // TODO: change rendering according to settings
        self.settings = settings;
    }

    pub fn register_update_handler(&mut self, handler: Box<UpdateHandler>) {
        self.update_handlers.push(handler);
    }

    pub fn load_texture_into_atlas(
        &mut self,
        texture_index: u32,
        atlas_type: TextureAtlasType,
        res_ref: ResourceRef,
    ) {
        let res_data = res_ref.read().unwrap();

        let mut t = basis_universal::Transcoder::new();
        t.prepare_transcoding(&res_data).unwrap();

        let img_info = t.image_info(&res_data, 0).unwrap();
        let width = img_info.m_width;
        let height = img_info.m_height;

        if !width.is_power_of_two() || width != height || width < (self.settings.texture_quality as u32) {
            return;
        }

        let mipmaps: Vec<_> = (0..img_info.m_total_levels)
            .map(|i| {
                t.transcode_image_level(
                    &res_data,
                    atlas_type.basis_decode_type(),
                    TranscodeParameters {
                        image_index: 0,
                        level_index: i,
                        decode_flags: None,
                        output_row_pitch_in_blocks_or_pixels: None,
                        output_rows_in_pixels: None,
                    },
                )
                .unwrap()
            })
            .collect();

        t.end_transcoding();

        let first_level = (width / (self.settings.texture_quality as u32)).ilog2();
        let last_level = (width / 4).ilog2(); // BC block size = 4x4

        self.res.texture_atlases[atlas_type as usize]
            .set_texture(
                texture_index,
                &mipmaps[(first_level as usize)..(last_level as usize + 1)],
            )
            .unwrap();
    }

    /// Returns id of registered material pipeline.
    pub fn register_material_pipeline(
        &mut self,
        shaders: &[Arc<Shader>],
        topology: PrimitiveTopology,
        cull_back_faces: bool,
    ) -> u32 {
        let main_signature = self
            .device
            .create_pipeline_signature(shaders, &*ADDITIONAL_PIPELINE_BINDINGS)
            .unwrap();

        let stages = self.stage_manager.stages();

        let mut per_frame_desc_pool = main_signature
            .create_pool(shader_ids::SET_CUSTOM_PER_FRAME, 1)
            .unwrap();
        let per_frame_desc = per_frame_desc_pool.alloc().unwrap();
        let per_object_desc_pool = main_signature
            .create_pool(shader_ids::SET_PER_OBJECT, 16)
            .unwrap();
        let mut pipeline_set = MaterialPipelineSet {
            device: Arc::clone(&self.device),
            main_signature: Arc::clone(&main_signature),
            pipelines: Default::default(),
            topology,
            per_object_desc_pool,
            per_frame_desc_pool,
            per_frame_desc,
        };

        for stage in stages.values() {
            stage.lock().register_pipeline_kind(
                MaterialPipelineParams {
                    shaders,
                    topology,
                    cull_back_faces,
                    main_signature: &main_signature,
                },
                &mut pipeline_set,
            );
        }

        self.res.material_pipelines.push(pipeline_set);
        (self.res.material_pipelines.len() - 1) as u32
    }

    pub fn get_material_pipeline(&self, id: u32) -> Option<&MaterialPipelineSet> {
        self.res.material_pipelines.get(id as usize)
    }

    pub fn get_material_pipeline_mut(&mut self, id: u32) -> Option<&mut MaterialPipelineSet> {
        self.res.material_pipelines.get_mut(id as usize)
    }

    pub fn set_material(&mut self, id: u32, info: MaterialInfo) {
        assert!(id < N_MAX_MATERIALS);
        self.material_updates.insert(id, info);
    }

    /// Copy each [u8] slice to appropriate DeviceBuffer with offset u64
    pub(crate) unsafe fn update_device_buffers(&mut self, updates: &[BufferUpdate]) {
        if updates.is_empty() {
            return;
        }

        let update_count = updates.len();
        let staging_size = self.staging_buffer.size();
        let mut used_size = 0;
        let mut i = 0;

        while i < update_count {
            let cl = self.staging_job.get_cmd_list_for_recording();
            cl.begin(true).unwrap();

            while i < update_count {
                let update = &updates[i];

                let (copy_size, new_used_size) = match update {
                    BufferUpdate::WithOffset(update) => {
                        let copy_size = update.data.len() as u64;
                        assert!(copy_size <= staging_size);
                        (copy_size, used_size + copy_size)
                    }
                    BufferUpdate::Regions(update) => {
                        let copy_size = update.data.len() as u64;
                        assert!(copy_size <= staging_size);
                        (copy_size, used_size + copy_size)
                    }
                };

                if new_used_size > staging_size {
                    used_size = 0;
                    break;
                }

                match update {
                    BufferUpdate::WithOffset(update) => {
                        self.staging_buffer.write(used_size as u64, &update.data);
                        cl.copy_buffer_to_device(
                            &self.staging_buffer,
                            used_size,
                            &update.buffer,
                            update.dst_offset,
                            copy_size,
                        );
                    }
                    BufferUpdate::Regions(update) => {
                        self.staging_buffer.write(used_size as u64, &update.data);

                        let regions: SmallVec<[CopyRegion; 64]> = update
                            .regions
                            .iter()
                            .map(|region| {
                                CopyRegion::new(
                                    used_size + region.src_offset(),
                                    region.dst_offset(),
                                    (region.size() as u64).try_into().unwrap(),
                                )
                            })
                            .collect();

                        cl.copy_buffer_regions_to_device_bytes(
                            &self.staging_buffer,
                            &update.buffer,
                            &regions,
                        );
                    }
                }

                used_size = new_used_size;
                i += 1;
            }

            cl.end().unwrap();

            self.device
                .run_jobs_sync(&mut [GPUJobExecInfo::new(&mut self.staging_job)])
                .unwrap();
        }
    }

    fn on_update(&mut self, ctx: &EngineContext) -> UpdateTimings {
        let mut timings = UpdateTimings::default();
        let total_t0 = Instant::now();

        let camera = *self.active_camera();
        let curr_rel_camera_pos = camera.position() - self.camera_pos_pivot;

        // Reset camera to origin (0, 0, 0) to save rendering precision
        // when camera position is too far (distance > 4096) from origin
        if curr_rel_camera_pos.magnitude() >= RESET_CAMERA_POS_THRESHOLD {
            let mut scene = ctx.module_mut::<Scene>();
            let mut root = scene.entry(&self.root_entity);

            let global_transform = root.get_mut::<TransformC>();
            global_transform.position -= curr_rel_camera_pos;

            self.relative_camera_pos = Vec3::default();
            self.camera_pos_pivot = *camera.position();
        } else {
            self.relative_camera_pos = glm::convert(curr_rel_camera_pos);
        }

        // --------------------------------------------------------------------

        let mut module_update_jobs = vec![];

        // Call registered update handlers
        for handler in &self.update_handlers {
            let job = handler(ctx);
            if let Some(job) = job {
                module_update_jobs.push(job.borrow_mut_owned());
            }
        }

        unsafe {
            self.device
                .run_jobs_sync(
                    &mut module_update_jobs
                        .iter_mut()
                        .map(|v| GPUJobExecInfo::new(v))
                        .collect::<Vec<_>>(),
                )
                .unwrap();
        }

        // --------------------------------------------------------------------

        let mut scene = ctx.module_mut::<Scene>();
        let mut change_manager = scene.change_manager_mut();

        let mut uniform_buffer_updates = BufferUpdate2 {
            buffer: self.res.uniform_buffer_basic.handle(),
            data: smallvec![],
            regions: vec![],
        };
        let mut buffer_updates = vec![];

        // Wait for transfer and acquire buffers from transfer queue to render queue
        if !self.vertex_mesh_pending_updates.is_empty() {
            unsafe {
                self.device
                    .run_jobs_sync(&mut [GPUJobExecInfo::new(&mut self.transfer_jobs.sync)
                        .with_wait_semaphores(&[self.transfer_jobs.work.wait_semaphore()])])
                    .unwrap();
            }
        }

        // Before updating new buffers, collect all the completed updates to commit them
        let mut completed_updates = self.vertex_mesh_pending_updates.clone();
        self.vertex_mesh_pending_updates.clear();

        let t00 = Instant::now();

        let mut renderer_events_system = system::RenderConfigComponentEvents {
            device: &self.device,
            renderables: &mut self.res.renderables,
            component_changes: change_manager.take(self.render_config_component_changes),
            buffer_updates: &mut buffer_updates,
            material_pipelines: &mut self.res.material_pipelines,
            uniform_buffer_basic: &self.res.uniform_buffer_basic,
            uniform_buffer_offsets: &mut self.res.uniform_buffer_offsets,
            run_time: 0.0,
        };
        let mut vertex_mesh_system = system::VertexMeshCompEvents {
            component_changes: change_manager.take(self.mesh_component_changes),
            curr_vertex_meshes: &mut self.res.curr_vertex_meshes,
            completed_updates: &mut completed_updates,
            new_buffer_updates: &mut self.new_vertex_mesh_updates,
            run_time: 0.0,
        };
        let mut hierarchy_propagation_system = system::HierarchyPropagation {
            root_entity: self.root_entity,
            dirty_components: change_manager.take_new(self.transform_component_changes),
            ordered_entities: &mut self.ordered_entities,
            changed_global_transforms: Vec::with_capacity(4096),
            run_time: 0.0,
        };

        scene.storage_mut().dispatch_par([
            System::new(&mut renderer_events_system).with_mut::<MeshRenderConfigC>(),
            System::new(&mut vertex_mesh_system).with_mut::<VertexMeshC>(),
            System::new(&mut hierarchy_propagation_system)
                .with::<Relation>()
                .with::<TransformC>()
                .with_mut::<GlobalTransformC>(),
        ]);

        timings.batch0_render_events = renderer_events_system.run_time;
        timings.batch0_vertex_meshes = vertex_mesh_system.run_time;
        timings.batch0_hierarchy_propag = hierarchy_propagation_system.run_time;

        let t11 = Instant::now();
        timings.systems_batch0 = (t11 - t00).as_secs_f64();

        let t00 = Instant::now();

        // Sort by distance to perform updates of the nearest vertex meshes first
        let mut sorted_buffer_updates_entities: Vec<_> = {
            // let transforms = self.scene.storage_read::<GlobalTransform>();
            let access = scene.storage_mut().access();
            let transforms = access.component::<GlobalTransformC>();
            self.new_vertex_mesh_updates
                .keys()
                .map(|v| {
                    if let Some(transform) = transforms.get(v) {
                        (
                            *v,
                            (glm::convert::<_, Vec3>(transform.position) - self.relative_camera_pos)
                                .magnitude_squared(),
                        )
                    } else {
                        (*v, f32::MAX)
                    }
                })
                .collect()
        };
        sorted_buffer_updates_entities.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // --------------------------------------------------------------------------------------------------

        let mut global_transform_events_system = system::GlobalTransformEvents {
            dirty_components: hierarchy_propagation_system.changed_global_transforms,
            changed_uniforms: Default::default(),
            run_time: 0.0,
        };
        global_transform_events_system.run(scene.storage_mut().access());

        for entity in &global_transform_events_system.changed_uniforms {
            change_manager.record_modification::<UniformDataC>(*entity);
        }

        // --------------------------------------------------------------------------------------------------

        let mut uniform_data_events_system = system::UniformDataCompEvents {
            uniform_buffer_updates: &mut uniform_buffer_updates,
            dirty_components: change_manager.take_new(self.uniform_data_component_changes),
            renderables: &self.res.renderables,
            run_time: 0.0,
        };
        let mut buffer_update_system = system::GpuBuffersUpdate {
            device: Arc::clone(&self.device),
            transfer_jobs: &mut self.transfer_jobs,
            buffer_updates: &mut self.new_vertex_mesh_updates,
            sorted_buffer_updates_entities: &sorted_buffer_updates_entities,
            pending_buffer_updates: &mut self.vertex_mesh_pending_updates,
            run_time: 0.0,
        };
        let mut commit_buffer_updates_system = system::CommitBufferUpdates {
            updates: completed_updates,
            vertex_meshes: &mut self.res.curr_vertex_meshes,
            run_time: 0.0,
        };

        scene.storage_mut().dispatch_par([
            System::new(&mut uniform_data_events_system).with::<UniformDataC>(),
            System::new(&mut buffer_update_system),
            System::new(&mut commit_buffer_updates_system).with::<VertexMeshC>(),
        ]);

        timings.batch1_global_transforms = global_transform_events_system.run_time;
        timings.batch1_buffer_updates = buffer_update_system.run_time;
        timings.batch1_updates_commit = commit_buffer_updates_system.run_time;

        let t11 = Instant::now();
        timings.systems_batch1 = (t11 - t00).as_secs_f64();

        // Update camera uniform buffers
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut per_frame_infos: [FrameInfoUniforms; 2] = Default::default();

            per_frame_infos[RenderType::MAIN as usize] = {
                let cam_pos: Vec3 = glm::convert(self.relative_camera_pos);
                let cam_dir = camera.direction();
                let proj = camera.projection();
                let view: Mat4 = glm::convert(camera::create_view_matrix(
                    &glm::convert(cam_pos),
                    &camera.rotation(),
                ));

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

                    atlas_info: U32Vec4::new(self.res.texture_atlases[0].tile_width(), 0, 0, 0),
                    frame_size: glm::convert_unchecked(self.render_size),
                }
            };
            per_frame_infos[RenderType::OVERLAY as usize] = {
                let cam_dir = self.overlay_camera.direction();
                let proj = self.overlay_camera.projection();
                let view = camera::create_view_matrix(
                    &glm::convert(*self.overlay_camera.position()),
                    self.overlay_camera.rotation(),
                );

                FrameInfoUniforms {
                    camera: CameraInfo {
                        pos: Vec4::new(0.0, 0.0, 0.0, 0.0),
                        dir: Vec4::new(cam_dir.x, cam_dir.y, cam_dir.z, 0.0),
                        proj,
                        view,
                        proj_view: proj * view,
                        z_near: camera.z_near(),
                        fovy: camera.fovy(),
                    },
                    ..per_frame_infos[RenderType::MAIN as usize]
                }
            };

            buffer_updates.extend(per_frame_infos.iter().enumerate().map(|(idx, info)| {
                BufferUpdate::WithOffset(BufferUpdate1 {
                    buffer: self.res.per_frame_ub.handle(),
                    dst_offset: idx as u64 * self.res.per_frame_ub.aligned_element_size(),
                    data: unsafe {
                        slice::from_raw_parts(info as *const _ as *const u8, mem::size_of_val(info))
                            .to_smallvec()
                    },
                })
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

            buffer_updates.push(BufferUpdate::Regions(BufferUpdate2 {
                buffer: self.res.material_buffer.handle(),
                data: data.to_smallvec(),
                regions,
            }));
        }

        buffer_updates.push(BufferUpdate::Regions(uniform_buffer_updates));

        drop(scene);
        drop(change_manager);
        unsafe { self.update_device_buffers(&buffer_updates) };

        let t1 = Instant::now();
        timings.uniform_buffers_update = (t1 - t11).as_secs_f64();

        for job in module_update_jobs {
            job.wait().unwrap();
        }

        let total_t1 = Instant::now();
        timings.total = (total_t1 - total_t0).as_secs_f64();

        timings
    }

    fn on_render(&mut self, sw_image: &SwapchainImage, ctx: &EngineContext) -> RenderTimings {
        let mut timings = RenderTimings::default();
        let t0 = Instant::now();

        let scene = ctx.module_mut::<Scene>();

        let stage_ctx = StageContext {
            storage: scene.storage(),
            material_pipelines: &self.res.material_pipelines,
            ordered_entities: &self.ordered_entities,
            active_camera: &self.active_camera,
            relative_camera_pos: self.relative_camera_pos,
            curr_vertex_meshes: &self.res.curr_vertex_meshes,
            renderables: &self.res.renderables,
            g_per_frame_pool: &self.res.g_per_frame_pool,
            g_per_frame_in: self.res.g_per_frame_in,
            per_frame_ub: &self.res.per_frame_ub,
            uniform_buffer_basic: &self.res.uniform_buffer_basic,
            render_size: (self.render_size.x, self.render_size.y),
            swapchain: self.swapchain.as_ref().unwrap(),
            render_sw_image: sw_image,
            frame_completion_semaphore: &self.frame_completion_semaphore,
        };

        unsafe {
            self.stage_manager.run(&stage_ctx).unwrap();
        }
        let t1 = Instant::now();

        timings.total = (t1 - t0).as_secs_f64();
        timings
    }

    pub fn on_draw(&mut self, ctx: &EngineContext) -> RendererTimings {
        let mut timings = RendererTimings::default();
        let device = Arc::clone(&self.device);
        let adapter = device.adapter();
        let surface = &self.surface;

        if !adapter.is_surface_valid(surface).unwrap() {
            return timings;
        }

        // Wait for previous frame completion
        self.stage_manager.wait_idle();

        if self.surface_changed {
            self.swapchain = Some(Arc::new(
                device
                    .create_swapchain(
                        &self.surface,
                        (self.surface_size.x as u32, self.surface_size.y as u32),
                        self.settings.fps_limit == FPSLimit::VSync,
                        if self.settings.prefer_triple_buffering {
                            3
                        } else {
                            2
                        },
                        None,
                        // self.swapchain.take(),
                    )
                    .unwrap(),
            ));

            self.surface_changed = false;
        }

        let acquire_result = self.swapchain.as_ref().unwrap().acquire_image();

        match acquire_result {
            Ok((sw_image, suboptimal)) => {
                self.surface_changed |= suboptimal;

                timings.update = self.on_update(ctx);
                timings.render = self.on_render(&sw_image, ctx);

                let present_queue = self.device.get_queue(QueueType::Present);
                let present_result = present_queue.present(&self.frame_completion_semaphore, sw_image);

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

        let new_frame_ts = Instant::now();
        let frame_delta = (new_frame_ts - self.last_frame_ts).as_secs_f64();

        if let FPSLimit::Limit(limit) = self.settings.fps_limit {
            let expected_dt = 1.0 / (limit as f64);
            let to_wait = (expected_dt - frame_delta).max(0.0);
            common::utils::high_precision_sleep(Duration::from_secs_f64(to_wait), Duration::from_micros(50));
        }

        timings
    }

    fn on_resize(&mut self, new_wsi_size: WSISize<u32>) {
        self.render_size = new_wsi_size.real();

        let surf_size = self.device.adapter().get_surface_size(&self.surface).unwrap();
        self.surface_size = glm::vec2(surf_size.0, surf_size.1);
        self.surface_changed = true;

        self.active_camera
            .set_aspect(self.render_size.x, self.render_size.y);
    }
}

impl EngineModule for MainRenderer {
    fn on_update(&mut self, _: f64, ctx: &EngineContext) {
        self.on_draw(ctx);
    }

    fn on_wsi_event(&mut self, _: &Window, event: &WSIEvent, _: &EngineContext) {
        match event {
            WSIEvent::Resized(new_size) => {
                self.on_resize(*new_size);
            }
            _ => {}
        }
    }
}

impl Drop for MainRenderer {
    fn drop(&mut self) {
        // Safe pipeline cache
        let pl_cache = self.device.get_pipeline_cache().unwrap();
        fs::write(*PIPELINE_CACHE_FILENAME, pl_cache).unwrap();

        self.device.wait_idle().unwrap();
    }
}
