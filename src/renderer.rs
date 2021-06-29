pub(crate) mod component;
#[macro_use]
pub(crate) mod material_pipeline;
pub mod material_pipelines;
mod texture_atlas;
#[macro_use]
pub(crate) mod vertex_mesh;
pub mod scene;
mod systems;

pub use scene::Scene;
pub use vertex_mesh::VertexMesh;

use crate::resource_file::{ResourceFile, ResourceRef};
use crate::utils;
use crate::utils::{HashMap, UInteger};
use ktx::KtxInfo;
use lazy_static::lazy_static;
use material_pipeline::{MaterialPipeline, PipelineMapping};
use nalgebra as na;
use nalgebra::{Matrix4, Vector4};
use nalgebra_glm as glm;
use rayon::prelude::*;
use scene::ComponentStorage;
use smallvec::SmallVec;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{iter, mem, slice};
use texture_atlas::TextureAtlas;
use vertex_mesh::VertexMeshCmdList;
use vk_wrapper::{
    swapchain, AccessFlags, Binding, BindingRes, BindingType, CopyRegion, DescriptorPool, DescriptorSet,
    DeviceError, HostBuffer, Image, ImageUsageFlags, ImageView, RawHostBuffer, Sampler, ShaderBinding,
    ShaderBindingMod, ShaderStage, SignalSemaphore, SwapchainImage,
};
use vk_wrapper::{
    Attachment, AttachmentRef, BufferUsageFlags, ClearValue, CmdList, Device, DeviceBuffer, Format,
    Framebuffer, ImageLayout, ImageMod, LoadStore, Pipeline, PipelineDepthStencil, PipelineRasterization,
    PipelineSignature, PipelineStageFlags, PrimitiveTopology, Queue, RenderPass, SubmitInfo, SubmitPacket,
    Subpass, Surface, Swapchain, WaitSemaphore,
};

pub struct Renderer {
    scene: Scene,

    surface: Arc<Surface>,
    swapchain: Option<Arc<Swapchain>>,
    surface_changed: bool,
    surface_size: (u32, u32),
    settings: Settings,
    device: Arc<Device>,

    texture_resources: Vec<(ResourceRef, TextureAtlasType, Option<u32>)>,
    texture_atlases: [TextureAtlas; 4],
    free_texture_indices: [Vec<u32>; 4],

    staging_buffer: HostBuffer<u8>,
    staging_cl: Arc<Mutex<CmdList>>,
    staging_submit: Arc<Mutex<SubmitPacket>>,
    final_cl: [Arc<Mutex<CmdList>>; 2],
    final_submit: [SubmitPacket; 2],

    sw_framebuffers: Vec<Arc<Framebuffer>>,

    depth_secondary_cls: Vec<Arc<Mutex<CmdList>>>,
    g_secondary_cls: Vec<Arc<Mutex<CmdList>>>,

    depth_render_pass: Arc<RenderPass>,
    depth_pyramid_image: Option<Arc<Image>>,
    depth_framebuffer: Option<Arc<Framebuffer>>,
    depth_signature: Arc<PipelineSignature>,
    depth_pipeline_r: Arc<Pipeline>,
    depth_pipeline_rw: Arc<Pipeline>,
    depth_per_object_pool: Option<DescriptorPool>,
    _depth_per_frame_pool: DescriptorPool,
    depth_per_frame_in: DescriptorSet,

    depth_pyramid_pipeline: Arc<Pipeline>,
    depth_pyramid_signature: Arc<PipelineSignature>,
    depth_pyramid_pool: Option<DescriptorPool>,
    depth_pyramid_descs: Vec<DescriptorSet>,
    depth_pyramid_sampler: Arc<Sampler>,

    cull_pipeline: Arc<Pipeline>,
    cull_signature: Arc<PipelineSignature>,
    cull_pool: DescriptorPool,
    cull_desc: DescriptorSet,
    cull_buffer: Arc<DeviceBuffer>,
    cull_host_buffer: HostBuffer<CullObject>,
    visibility_buffer: Arc<DeviceBuffer>,
    visibility_host_buffer: HostBuffer<u32>,

    sw_render_pass: Option<Arc<RenderPass>>,
    compose_pipeline: Option<Arc<Pipeline>>,
    compose_signature: Arc<PipelineSignature>,
    compose_pool: DescriptorPool,
    compose_desc: DescriptorSet,

    g_render_pass: Arc<RenderPass>,
    g_framebuffer: Option<Arc<Framebuffer>>,
    _g_per_frame_pool: DescriptorPool,
    g_per_frame_desc: DescriptorSet,
    g_per_pipeline_pools: Option<HashMap<Arc<PipelineSignature>, DescriptorPool>>,

    translucency_head_image: Option<Arc<Image>>,
    translucency_texel_image: Option<Arc<Image>>,

    active_camera_desc: u32,
    per_frame_ub: Arc<DeviceBuffer>,
    material_buffer: Arc<DeviceBuffer>,
    material_updates: HashMap<u32, MaterialInfo>,

    renderables: Arc<Mutex<HashMap<u32, Renderable>>>,
}

#[derive(Copy, Clone)]
pub enum TextureQuality {
    STANDARD = 128,
    HIGH = 256,
}

#[derive(Copy, Clone)]
pub enum TranslucencyMaxDepth {
    LOW = 4,
    MEDIUM = 8,
    HIGH = 16,
}

#[derive(Copy, Clone)]
pub struct Settings {
    pub(crate) vsync: bool,
    pub(crate) texture_quality: TextureQuality,
    pub(crate) translucency_max_depth: TranslucencyMaxDepth,
    pub(crate) textures_gen_mipmaps: bool,
    pub(crate) textures_max_anisotropy: f32,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub enum TextureAtlasType {
    ALBEDO = 0,
    SPECULAR = 1,
    EMISSION = 2,
    NORMAL = 3,
}

struct BufferUpdate1 {
    buffer: Arc<DeviceBuffer>,
    offset: u64,
    data: Vec<u8>,
}

struct BufferUpdate2 {
    src_buffer: RawHostBuffer,
    src_offset: u64,
    dst_buffer: Arc<DeviceBuffer>,
    dst_offset: u64,
    size: u64,
}

struct BufferUpdate3 {
    buffer: Arc<DeviceBuffer>,
    data: Vec<u8>,
    // (src data offset, dst offset, size)
    regions: Vec<(u64, u64, u64)>,
}

enum BufferUpdate {
    Type1(BufferUpdate1),
    Type2(BufferUpdate2),
    Type3(BufferUpdate3),
}

struct Renderable {
    buffers: SmallVec<[Arc<DeviceBuffer>; 4]>,
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
    _pad: [f32; 2],
}

#[derive(Debug)]
#[repr(C)]
struct PerFrameInfo {
    camera: CameraInfo,
    atlas_info: na::Vector4<u32>,
}

#[repr(C)]
pub struct MaterialInfo {
    pub(crate) diffuse_tex_id: u32,
    pub(crate) specular_tex_id: u32,
    pub(crate) normal_tex_id: u32,
    pub _pad: u32,
    pub(crate) diffuse: na::Vector4<f32>,
    pub(crate) specular: na::Vector4<f32>,
    pub(crate) emission: na::Vector4<f32>,
}

#[repr(C)]
struct DepthPyramidConstants {
    size: na::Vector2<f32>,
}

#[repr(C)]
struct CullObject {
    sphere: na::Vector4<f32>,
    id: u32,
    _pad: [u32; 3],
}

#[repr(C)]
struct CullConstants {
    pyramid_size: na::Vector2<f32>,
    object_count: u32,
}

const MAX_OBJECT_COUNT: u32 = 65535;
const MAX_MATERIAL_COUNT: u32 = 4096;
const COMPUTE_LOCAL_THREADS: u32 = 32;

lazy_static! {
static ref ADDITIONAL_PIPELINE_BINDINGS: [(ShaderStage, &'static [ShaderBinding]); 3] = [
    (
        ShaderStage::VERTEX,
        &[
            // Per object info
            ShaderBinding {
                binding_type: BindingType::UNIFORM_BUFFER,
                binding_mod: ShaderBindingMod::DEFAULT,
                descriptor_set: 1,
                id: 0,
                count: 1,
            },
        ],
    ),
    (
        ShaderStage::VERTEX | ShaderStage::PIXEL,
        &[
            // Per frame info
            ShaderBinding {
                binding_type: BindingType::UNIFORM_BUFFER,
                binding_mod: ShaderBindingMod::DEFAULT,
                descriptor_set: 0,
                id: 0,
                count: 1,
            },
        ],
    ),
    (
        ShaderStage::PIXEL,
        &[
            // Albedo atlas
            ShaderBinding {
                binding_type: BindingType::SAMPLED_IMAGE,
                binding_mod: ShaderBindingMod::DEFAULT,
                descriptor_set: 0,
                id: 1,
                count: 1,
            },
            // Specular atlas
            ShaderBinding {
                binding_type: BindingType::SAMPLED_IMAGE,
                binding_mod: ShaderBindingMod::DEFAULT,
                descriptor_set: 0,
                id: 2,
                count: 1,
            },
            // Normal atlas
            ShaderBinding {
                binding_type: BindingType::SAMPLED_IMAGE,
                binding_mod: ShaderBindingMod::DEFAULT,
                descriptor_set: 0,
                id: 3,
                count: 1,
            },
            // Material buffer
            ShaderBinding {
                binding_type: BindingType::STORAGE_BUFFER,
                binding_mod: ShaderBindingMod::DEFAULT,
                descriptor_set: 0,
                id: 4,
                count: 1,
            },
        ],
    ),
];
}

fn calc_group_count(thread_count: u32) -> u32 {
    (thread_count + COMPUTE_LOCAL_THREADS - 1) / COMPUTE_LOCAL_THREADS
}

impl Renderer {
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn get_active_camera(&self) -> u32 {
        self.active_camera_desc
    }

    pub fn set_active_camera(&mut self, entity: u32) {
        self.active_camera_desc = entity;
    }

    /// Add texture to renderer
    pub fn add_texture(&mut self, atlas_type: TextureAtlasType, res_ref: ResourceRef) -> usize {
        self.texture_resources.push((res_ref, atlas_type, None));
        self.texture_resources.len() - 1
    }

    /// Texture must be loaded before use in a shader
    pub fn load_texture(&mut self, index: usize) {
        let (res_ref, atlas_type, tex_index) = &mut self.texture_resources[index];

        if tex_index.is_none() {
            let free_indices = &mut self.free_texture_indices[*atlas_type as usize];
            *tex_index = if free_indices.is_empty() {
                None
            } else {
                Some(free_indices.swap_remove(0))
            };
        }

        if let Some(tex_index) = tex_index {
            let res_data = res_ref.read().unwrap();

            let decoder = ktx::Decoder::new(res_data.as_slice()).unwrap();
            let width = decoder.pixel_width();
            let height = decoder.pixel_height();

            if !utils::is_pow_of_2(width as u64)
                || width != height
                || width < (self.settings.texture_quality as u32)
            {
                return;
            }

            let mip_maps: Vec<Vec<u8>> = decoder.read_textures().collect();

            let first_level = (width / (self.settings.texture_quality as u32)).log2();
            let last_level = (width / 4).log2(); // BC block size = 4x4

            self.texture_atlases[*atlas_type as usize]
                .set_texture(
                    *tex_index,
                    &mip_maps[(first_level as usize)..(last_level as usize + 1)],
                )
                .unwrap();
        }
    }

    /// Unload unused texture to free GPU memory for another texture
    pub fn unload_texture(&mut self, index: u32) {
        let (_res_ref, atlas_type, tex_index) = &mut self.texture_resources[index as usize];

        if let Some(tex_index) = tex_index {
            self.free_texture_indices[*atlas_type as usize].push(*tex_index);
        }
        *tex_index = None;
    }

    pub fn prepare_material_pipeline(&mut self, material_pipeline: &mut MaterialPipeline) {
        material_pipeline.prepare_pipeline(&PipelineMapping {
            render_pass: Arc::clone(&self.g_render_pass),
            subpass_index: 0,
            cull_back_faces: false,
        });
        material_pipeline.prepare_pipeline(&PipelineMapping {
            render_pass: Arc::clone(&self.g_render_pass),
            subpass_index: 0,
            cull_back_faces: true,
        });

        let g_per_pipeline_pools = self.g_per_pipeline_pools.as_mut().unwrap();
        let signature = material_pipeline.signature();

        if !g_per_pipeline_pools.contains_key(signature) {
            g_per_pipeline_pools.insert(Arc::clone(signature), signature.create_pool(1, 16).unwrap());
        }
    }

    pub fn set_material(&mut self, id: u32, info: MaterialInfo) {
        self.material_updates.insert(id, info);
    }

    /// Copy each [u8] slice to appropriate DeviceBuffer with offset u64
    unsafe fn update_device_buffers(&mut self, updates: &[BufferUpdate]) {
        if updates.is_empty() {
            return;
        }

        let update_count = updates.len();

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

        let staging_size = self.staging_buffer.size();
        let mut used_size = 0;
        let mut i = 0;

        while i < update_count {
            {
                let mut cl = self.staging_cl.lock().unwrap();
                cl.begin(true).unwrap();

                while i < update_count {
                    let update = &updates[i];

                    let (copy_size, new_used_size) = match update {
                        BufferUpdate::Type1(update) => {
                            let copy_size = update.data.len() as u64;
                            assert!(copy_size <= staging_size);
                            (copy_size, used_size + copy_size)
                        }
                        BufferUpdate::Type2(update) => (update.size, used_size),
                        BufferUpdate::Type3(update) => {
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
                        BufferUpdate::Type1(update) => {
                            self.staging_buffer.write(used_size as u64, &update.data);

                            cl.copy_buffer_to_device(
                                &self.staging_buffer,
                                used_size,
                                &update.buffer,
                                update.offset,
                                copy_size,
                            );
                        }
                        BufferUpdate::Type2(update) => {
                            cl.copy_raw_host_buffer_to_device(
                                &update.src_buffer,
                                update.src_offset,
                                &update.dst_buffer,
                                update.dst_offset,
                                update.size,
                            );
                        }
                        BufferUpdate::Type3(update) => {
                            self.staging_buffer.write(used_size as u64, &update.data);

                            let regions: SmallVec<[CopyRegion; 128]> = update
                                .regions
                                .iter()
                                .map(|region| CopyRegion {
                                    src_element_index: used_size + region.0,
                                    dst_element_index: region.1,
                                    size: region.2,
                                })
                                .collect();

                            cl.copy_buffer_regions_to_device(&self.staging_buffer, &update.buffer, &regions);
                        }
                    }

                    used_size = new_used_size;
                    i += 1;
                }

                cl.end().unwrap();
            }

            let mut submit = self.staging_submit.lock().unwrap();
            graphics_queue.submit(&mut submit).unwrap();
            submit.wait().unwrap();
        }
    }

    pub fn set_settings(&mut self, settings: Settings) {
        // TODO
        self.settings = settings;
    }

    pub fn on_update(&mut self) {
        let camera = {
            let camera_comps = self.scene.storage::<component::Camera>();
            let camera_comps = camera_comps.read().unwrap();
            *camera_comps.get(self.get_active_camera()).unwrap()
        };
        let buffer_updates = Arc::new(Mutex::new(vec![]));
        let buffer_updates2 = Arc::new(Mutex::new(vec![]));
        let buffer_updates3 = Arc::new(Mutex::new(vec![]));

        let mut renderer_events_system = systems::RendererCompEventsSystem {
            renderer_comps: self.scene.storage::<component::Renderer>(),
            depth_per_object_pool: self.depth_per_object_pool.take().unwrap(),
            g_per_pipeline_pools: self.g_per_pipeline_pools.take().unwrap(),
            renderables: Arc::clone(&self.renderables),
            buffer_updates: Arc::clone(&buffer_updates),
        };

        let mut vertex_mesh_system = systems::VertexMeshCompEventsSystem {
            vertex_mesh_comps: self.scene.storage::<component::VertexMesh>(),
            buffer_updates: Arc::clone(&buffer_updates2),
        };

        let mut transform_events_system = systems::TransformEventsSystem {
            transform_comps: self.scene.storage::<component::Transform>(),
            model_transform_comps: self.scene.storage::<component::ModelTransform>(),
        };

        let mut world_transform_events_system = systems::WorldTransformEventsSystem {
            buffer_updates: Arc::clone(&buffer_updates3),
            world_transform_comps: self.scene.storage::<component::WorldTransform>(),
            renderer_comps: self.scene.storage::<component::Renderer>(),
        };

        let mut hierarchy_propagation_system = systems::HierarchyPropagationSystem {
            parent_comps: self.scene.storage::<component::Parent>(),
            children_comps: self.scene.storage::<component::Children>(),
            model_transform_comps: self.scene.storage::<component::ModelTransform>(),
            world_transform_comps: self.scene.storage::<component::WorldTransform>(),
        };

        rayon::scope(|_| {
            rayon::scope(|s| {
                s.spawn(|_| {
                    renderer_events_system.run();
                });
                s.spawn(|_| {
                    vertex_mesh_system.run();
                });
                s.spawn(|_| {
                    transform_events_system.run();
                });
            });
            rayon::scope(|_| {
                hierarchy_propagation_system.run();
            });
            rayon::scope(|s| {
                s.spawn(|_| {
                    world_transform_events_system.run();
                });
            });
        });

        self.depth_per_object_pool = Some(renderer_events_system.depth_per_object_pool);
        self.g_per_pipeline_pools = Some(renderer_events_system.g_per_pipeline_pools);

        // Update camera uniform buffers
        // -------------------------------------------------------------------------------------------------------------
        {
            let per_frame_info = {
                let cam_pos = camera.position();
                let cam_dir = camera.direction();
                let proj = camera.projection();
                let view = camera.view();

                PerFrameInfo {
                    camera: CameraInfo {
                        pos: Vector4::new(cam_pos.x, cam_pos.y, cam_pos.z, 0.0),
                        dir: Vector4::new(cam_dir.x, cam_dir.y, cam_dir.z, 0.0),
                        proj,
                        view,
                        proj_view: proj * view,
                        z_near: camera.z_near(),
                        fovy: camera.fovy(),
                        _pad: [0.0; 2],
                    },
                    atlas_info: na::Vector4::new(self.texture_atlases[0].tile_width(), 0, 0, 0),
                }
            };

            let data = unsafe {
                slice::from_raw_parts(
                    &per_frame_info as *const PerFrameInfo as *const u8,
                    mem::size_of_val(&per_frame_info),
                )
                .to_vec()
            };
            buffer_updates
                .lock()
                .unwrap()
                .push(BufferUpdate::Type1(BufferUpdate1 {
                    buffer: Arc::clone(&self.per_frame_ub),
                    offset: 0,
                    data,
                }));
        }

        // Update material buffer
        // -------------------------------------------------------------------------------------------------------------
        if !self.material_updates.is_empty() {
            let mut data = Vec::<MaterialInfo>::with_capacity(self.material_updates.len());
            let mat_size = mem::size_of::<MaterialInfo>() as u64;

            let regions: Vec<(u64, u64, u64)> = self
                .material_updates
                .drain()
                .enumerate()
                .map(|(i, info)| {
                    data.push(info.1);
                    (i as u64 * mat_size, info.0 as u64 * mat_size, mat_size)
                })
                .collect();

            let data =
                unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * mat_size as usize) };

            buffer_updates
                .lock()
                .unwrap()
                .push(BufferUpdate::Type3(BufferUpdate3 {
                    buffer: Arc::clone(&self.material_buffer),
                    data: data.to_vec(),
                    regions,
                }));

            unsafe { self.update_device_buffers(&[]) };
        }

        // Wait for vertex buffer updates
        self.staging_submit.lock().unwrap().wait().unwrap();

        unsafe {
            self.update_device_buffers(&buffer_updates.lock().unwrap());
            self.update_device_buffers(&buffer_updates2.lock().unwrap());
            self.update_device_buffers(&buffer_updates3.lock().unwrap());
        }

        // let t2 = Instant::now();
        // println!("{}", (t2 - t).as_secs_f64());
    }

    fn record_depth_cmd_lists(
        &mut self,
        renderables: &[u32],
        camera: &component::Camera,
        world_transform_comps: &ComponentStorage<component::WorldTransform>,
        renderer_comps: &ComponentStorage<component::Renderer>,
        vertex_mesh_comps: &ComponentStorage<component::VertexMesh>,
    ) -> u32 {
        let object_count = renderables.len();
        let draw_count_step = object_count / self.depth_secondary_cls.len() + 1;

        let cull_objects = Mutex::new(Vec::<CullObject>::with_capacity(object_count));

        self.depth_secondary_cls
            .par_iter()
            .enumerate()
            .for_each(|(i, cmd_list)| {
                let mut curr_cull_objects = Vec::with_capacity(draw_count_step);

                let mut cl = cmd_list.lock().unwrap();

                cl.begin_secondary_graphics(
                    true,
                    &self.depth_render_pass,
                    0,
                    Some(self.depth_framebuffer.as_ref().unwrap()),
                )
                .unwrap();

                cl.bind_graphics_input(&self.depth_signature, 0, self.depth_per_frame_in);

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable = renderables[entity_index];

                    let transform = world_transform_comps.get(renderable);
                    let renderer = renderer_comps.get(renderable);
                    let vertex_mesh = vertex_mesh_comps.get(renderable);

                    if transform.is_none() || renderer.is_none() || vertex_mesh.is_none() {
                        continue;
                    }

                    let transform = transform.unwrap();
                    let renderer = renderer.unwrap();
                    let vertex_mesh = vertex_mesh.unwrap();

                    let vertex_mesh = &vertex_mesh.0;

                    if vertex_mesh.vertex_count == 0 {
                        continue;
                    }

                    let sphere = vertex_mesh.sphere();
                    let center = sphere.center() + transform.position;
                    let radius = sphere.radius() * glm::comp_max(&transform.scale);

                    if !camera.is_sphere_visible(&center, radius) {
                        continue;
                    }

                    curr_cull_objects.push(CullObject {
                        sphere: na::Vector4::new(center.x, center.y, center.z, radius),
                        id: entity_index as u32,
                        _pad: [0; 3],
                    });

                    if renderer.translucent {
                        cl.bind_pipeline(&self.depth_pipeline_r);
                    } else {
                        cl.bind_pipeline(&self.depth_pipeline_rw);
                    }

                    cl.bind_graphics_input(&self.depth_signature, 1, renderer.descriptor_sets[0]);
                    cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                }

                cl.end().unwrap();

                cull_objects.lock().unwrap().extend(curr_cull_objects);
            });

        let cull_objects = cull_objects.into_inner().unwrap();
        let count = cull_objects.len() as u32;

        cull_objects.into_iter().enumerate().for_each(|(i, obj)| {
            self.cull_host_buffer[i] = obj;
        });

        count
    }

    fn record_g_cmd_lists(
        &self,
        renderables: &[u32],
        renderer_comps: &ComponentStorage<component::Renderer>,
        vertex_mesh_comps: &ComponentStorage<component::VertexMesh>,
    ) {
        let object_count = renderables.len();
        let draw_count_step = object_count / self.g_secondary_cls.len() + 1;

        let pipeline_mapping = PipelineMapping {
            render_pass: Arc::clone(&self.g_render_pass),
            subpass_index: 0,
            cull_back_faces: true,
        };

        self.g_secondary_cls
            .par_iter()
            .enumerate()
            .for_each(|(i, cmd_list)| {
                let mut cl = cmd_list.lock().unwrap();

                cl.begin_secondary_graphics(
                    true,
                    &self.g_render_pass,
                    0,
                    Some(self.g_framebuffer.as_ref().unwrap()),
                )
                .unwrap();

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let renderable = renderables[entity_index];

                    let renderer = renderer_comps.get(renderable).unwrap();
                    let vertex_mesh = vertex_mesh_comps.get(renderable);

                    if vertex_mesh.is_none() {
                        continue;
                    }

                    // Check query_pool occlusion results
                    if self.visibility_host_buffer[entity_index] == 0 {
                        continue;
                    }

                    let mesh = vertex_mesh.unwrap();
                    let vertex_mesh = &mesh.0;

                    let mat_pipeline = &renderer.mat_pipeline;
                    let pipeline = mat_pipeline.get_pipeline(&pipeline_mapping).unwrap();
                    let signature = pipeline.signature();

                    let already_bound = cl.bind_pipeline(pipeline);
                    if !already_bound {
                        cl.bind_graphics_input(&signature, 0, self.g_per_frame_desc);
                    }
                    cl.bind_graphics_input(&signature, 1, renderer.descriptor_sets[1]);
                    cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                }

                cl.end().unwrap();
            });
    }

    fn on_render(&mut self, sw_image: &SwapchainImage) {
        let device = Arc::clone(&self.device);
        let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);

        let camera = {
            let camera_comps = self.scene.storage::<component::Camera>();
            let camera_comps = camera_comps.read().unwrap();
            *camera_comps.get(self.get_active_camera()).unwrap()
        };

        let world_transform_comp = self.scene.storage::<component::WorldTransform>();
        let world_transform_comps = world_transform_comp.read().unwrap();
        let renderer_comp = self.scene.storage::<component::Renderer>();
        let renderer_comps = renderer_comp.read().unwrap();
        let vertex_mesh_comp = self.scene.storage::<component::VertexMesh>();
        let vertex_mesh_comps = vertex_mesh_comp.read().unwrap();

        let renderables: Vec<u32> = renderer_comps.entries().iter().map(|v| v as u32).collect();
        let object_count = renderables.len() as u32;

        let frustum_visible_objects = self.record_depth_cmd_lists(
            &renderables,
            &camera,
            &world_transform_comps,
            &renderer_comps,
            &vertex_mesh_comps,
        );

        {
            let mut cl = self.staging_cl.lock().unwrap();
            cl.begin(true).unwrap();
            cl.begin_render_pass(
                &self.depth_render_pass,
                self.depth_framebuffer.as_ref().unwrap(),
                &[ClearValue::Depth(1.0)],
                true,
            );
            cl.execute_secondary(&self.depth_secondary_cls);
            cl.end_render_pass();

            let depth_image = self.depth_framebuffer.as_ref().unwrap().get_image(0).unwrap();
            let depth_pyramid_image = self.depth_pyramid_image.as_ref().unwrap();

            // Build depth pyramid
            // ------------------------------------------------------------------
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

            for i in 0..(depth_pyramid_image.mip_levels() as usize) {
                let size = depth_pyramid_image.size_2d();
                let level_width = (size.0 >> i).max(1);
                let level_height = (size.1 >> i).max(1);

                cl.bind_compute_input(&self.depth_pyramid_signature, 0, self.depth_pyramid_descs[i]);

                let constants = DepthPyramidConstants {
                    size: na::Vector2::new(level_width as f32, level_height as f32),
                };
                cl.push_constants(&self.depth_pyramid_signature, &constants);

                cl.dispatch(calc_group_count(level_width), calc_group_count(level_height), 1);

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
            }

            cl.barrier_image(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::BOTTOM_OF_PIPE,
                &[depth_image
                    .barrier()
                    .src_access_mask(AccessFlags::SHADER_READ)
                    .dst_access_mask(Default::default())
                    .old_layout(ImageLayout::SHADER_READ)
                    .new_layout(ImageLayout::DEPTH_READ)],
            );

            // Compute visibilities
            // ------------------------------------------------------------------
            cl.copy_buffer_to_device(
                &self.cull_host_buffer,
                0,
                &self.cull_buffer,
                0,
                frustum_visible_objects as u64,
            );
            cl.clear_buffer(&self.visibility_buffer, 0);

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
            cl.bind_compute_input(&self.cull_signature, 0, self.cull_desc);

            let pyramid_size = depth_pyramid_image.size_2d();
            let constants = CullConstants {
                pyramid_size: na::Vector2::new(pyramid_size.0 as f32, pyramid_size.1 as f32),
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
        }
        {
            let mut submit = self.staging_submit.lock().unwrap();
            unsafe {
                graphics_queue.submit(&mut submit).unwrap();
            }
            submit.wait().unwrap();
        }

        self.record_g_cmd_lists(&renderables, &renderer_comps, &vertex_mesh_comps);

        let albedo = self.g_framebuffer.as_ref().unwrap().get_image(0).unwrap();
        self.compose_pool.update(
            self.compose_desc,
            &[Binding {
                id: 0,
                array_index: 0,
                res: BindingRes::Image(Arc::clone(albedo), ImageLayout::SHADER_READ),
            }],
        );
        let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);

        // Record G-Buffer cmd list
        // -------------------------------------------------------------------------------------------------------------
        {
            // Note: Do not render anything in final cl except copying some image into swapchain image.
            // Uniform/vertex  may be being updated at this moment.
            let mut cl = self.final_cl[0].lock().unwrap();
            cl.begin(true).unwrap();

            // let _translucency_head_image = self.translucency_head_image.as_ref().unwrap();

            // TODO: translucency
            /*cl.barrier_image(
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                &[translucency_head_image.barrier_queue(
                    AccessFlags::default(),
                    AccessFlags::TRANSFER_WRITE,
                    ImageLayout::UNDEFINED,
                    ImageLayout::GENERAL,
                    graphics_queue,
                    graphics_queue,
                )],
            );
            cl.clear_image(
                translucency_head_image,
                ImageLayout::GENERAL,
                ClearValue::ColorU32([0xffffffff; 4]),
            );*/

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
            cl.execute_secondary(&self.g_secondary_cls);
            cl.end_render_pass();

            cl.barrier_image(
                PipelineStageFlags::ALL_GRAPHICS,
                PipelineStageFlags::ALL_GRAPHICS,
                &[albedo
                    .barrier()
                    .src_access_mask(AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(AccessFlags::MEMORY_READ)
                    .old_layout(ImageLayout::SHADER_READ)
                    .new_layout(ImageLayout::SHADER_READ)],
            );

            cl.begin_render_pass(
                self.sw_render_pass.as_ref().unwrap(),
                &self.sw_framebuffers[sw_image.get_index() as usize],
                &[],
                false,
            );
            cl.bind_pipeline(self.compose_pipeline.as_ref().unwrap());
            cl.bind_graphics_input(&self.compose_signature, 0, self.compose_desc);
            cl.draw(3, 0);
            cl.end_render_pass();

            if graphics_queue != present_queue {
                cl.barrier_image(
                    PipelineStageFlags::ALL_GRAPHICS,
                    PipelineStageFlags::BOTTOM_OF_PIPE,
                    &[sw_image
                        .get_image()
                        .barrier()
                        .src_access_mask(AccessFlags::COLOR_ATTACHMENT_WRITE)
                        .old_layout(ImageLayout::PRESENT)
                        .new_layout(ImageLayout::PRESENT)
                        .src_queue(graphics_queue)
                        .dst_queue(present_queue)],
                );
            }

            cl.end().unwrap();
        }

        unsafe {
            graphics_queue.submit(&mut self.final_submit[0]).unwrap();

            if graphics_queue != present_queue {
                {
                    let mut cl = self.final_cl[1].lock().unwrap();
                    cl.begin(true).unwrap();
                    cl.barrier_image(
                        PipelineStageFlags::TOP_OF_PIPE,
                        PipelineStageFlags::BOTTOM_OF_PIPE,
                        &[sw_image
                            .get_image()
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
                            semaphore: Arc::clone(present_queue.frame_semaphore()),
                            signal_value: 0,
                        }],
                    )])
                    .unwrap();

                present_queue.submit(&mut self.final_submit[1]).unwrap();
                self.final_submit[1].wait().unwrap();
            }
        }
    }

    pub fn on_draw(&mut self) {
        let device = Arc::clone(&self.device);
        let adapter = device.get_adapter();
        let surface = &self.surface;

        if adapter.is_surface_valid(surface).unwrap() {
            if self.surface_changed {
                let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
                let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);

                self.swapchain = Some(
                    device
                        .create_swapchain(
                            &self.surface,
                            self.surface_size,
                            self.settings.vsync,
                            self.swapchain.as_ref(),
                        )
                        .unwrap(),
                );

                let signal_sem = &[SignalSemaphore {
                    semaphore: Arc::clone(present_queue.frame_semaphore()),
                    signal_value: 0,
                }];

                self.final_submit[0]
                    .set(&[SubmitInfo::new(
                        &[WaitSemaphore {
                            semaphore: Arc::clone(self.swapchain.as_ref().unwrap().get_semaphore()),
                            wait_dst_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, // TODO: change if necessary
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

            let swapchain = Arc::clone(self.swapchain.as_ref().unwrap());
            let acquire_result = swapchain.acquire_image();

            match acquire_result {
                Ok((sw_image, suboptimal)) => {
                    self.surface_changed |= suboptimal;

                    // Note: wait for render completion before on_update()
                    // to not destroy DeviceBuffers after entity deletion in on_update()
                    self.final_submit[0].wait().unwrap();
                    self.final_submit[1].wait().unwrap();
                    self.on_update();
                    self.on_render(&sw_image);

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
        }
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_size = new_size;
        self.surface_changed = true;

        self.device.wait_idle().unwrap();

        // Set camera aspect
        {
            let entity = self.get_active_camera();
            let camera_comps = self.scene.storage::<component::Camera>();
            let mut camera_comps = camera_comps.write().unwrap();
            let camera = camera_comps.get_mut(entity).unwrap();
            camera.set_aspect(new_size.0, new_size.1);
        }

        let depth_image = self
            .device
            .create_image_2d(
                Format::D32_FLOAT,
                1,
                1.0,
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | ImageUsageFlags::SAMPLED,
                new_size,
            )
            .unwrap();
        self.depth_pyramid_image = Some(
            self.device
                .create_image_2d(
                    Format::R32_FLOAT,
                    0,
                    1.0,
                    ImageUsageFlags::SAMPLED | ImageUsageFlags::STORAGE,
                    // Note: prev_power_of_two makes sure all reductions are at most by 2x2
                    // which makes sure they are conservative
                    (
                        utils::prev_power_of_two(new_size.0),
                        utils::prev_power_of_two(new_size.1),
                    ),
                )
                .unwrap(),
        );
        let depth_pyramid_levels = self.depth_pyramid_image.as_ref().unwrap().mip_levels();
        let depth_pyramid_views: Vec<Arc<ImageView>> = (0..depth_pyramid_levels)
            .map(|i| {
                self.depth_pyramid_image
                    .as_ref()
                    .unwrap()
                    .create_view()
                    .base_mip_level(i)
                    .mip_level_count(1)
                    .build()
                    .unwrap()
            })
            .collect();

        self.depth_pyramid_pool = Some(
            self.depth_pyramid_signature
                .create_pool(0, depth_pyramid_levels)
                .unwrap(),
        );
        self.depth_pyramid_descs = {
            let sampler = &self.depth_pyramid_sampler;
            let pool = self.depth_pyramid_pool.as_mut().unwrap();

            (0..depth_pyramid_levels as usize)
                .map(|i| {
                    let id = pool.alloc().unwrap();
                    pool.update(
                        id,
                        &[
                            Binding {
                                id: 0,
                                array_index: 0,
                                res: if i == 0 {
                                    BindingRes::ImageViewSampler(
                                        Arc::clone(depth_image.view()),
                                        Arc::clone(&sampler),
                                        ImageLayout::SHADER_READ,
                                    )
                                } else {
                                    BindingRes::ImageViewSampler(
                                        Arc::clone(&depth_pyramid_views[i - 1]),
                                        Arc::clone(&sampler),
                                        ImageLayout::GENERAL,
                                    )
                                },
                            },
                            Binding {
                                id: 1,
                                array_index: 0,
                                res: BindingRes::ImageView(
                                    Arc::clone(&depth_pyramid_views[i]),
                                    ImageLayout::GENERAL,
                                ),
                            },
                        ],
                    );
                    id
                })
                .collect()
        };

        self.cull_pool.update(
            self.cull_desc,
            &[
                Binding {
                    id: 0,
                    array_index: 0,
                    res: BindingRes::ImageViewSampler(
                        Arc::clone(self.depth_pyramid_image.as_ref().unwrap().view()),
                        Arc::clone(&self.depth_pyramid_sampler),
                        ImageLayout::GENERAL,
                    ),
                },
                Binding {
                    id: 1,
                    array_index: 0,
                    res: BindingRes::Buffer(Arc::clone(&self.per_frame_ub)),
                },
                Binding {
                    id: 2,
                    array_index: 0,
                    res: BindingRes::Buffer(Arc::clone(&self.cull_buffer)),
                },
                Binding {
                    id: 3,
                    array_index: 0,
                    res: BindingRes::Buffer(Arc::clone(&self.visibility_buffer)),
                },
            ],
        );

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
                                    .create_image_2d(
                                        Format::R32_UINT,
                                        1,
                                        1.0,
                                        ImageUsageFlags::STORAGE,
                                        new_size,
                                    )
                                    .unwrap(),
                            ),
                        ),
                    ],
                )
                .unwrap(),
        );

        self.translucency_head_image = Some(
            self.device
                .create_image_2d(Format::R32_UINT, 1, 1.0, ImageUsageFlags::STORAGE, new_size)
                .unwrap(),
        );

        self.translucency_texel_image = Some(
            self.device
                .create_image_3d(
                    Format::RG32_UINT, // color & depth
                    ImageUsageFlags::STORAGE,
                    (
                        new_size.0,
                        new_size.1,
                        self.settings.translucency_max_depth as u32,
                    ),
                )
                .unwrap(),
        );
    }

    fn create_main_framebuffers(&mut self) {
        self.sw_framebuffers.clear();

        let images = self.swapchain.as_ref().unwrap().get_images();

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
                    &self.compose_signature,
                )
                .unwrap(),
        );
    }
}

pub fn new(
    surface: &Arc<Surface>,
    size: (u32, u32),
    settings: Settings,
    device: &Arc<Device>,
    resources: &Arc<ResourceFile>,
    max_texture_count: u32,
) -> Result<Arc<Mutex<Renderer>>, DeviceError> {
    let scene = Scene::new();

    // TODO: pipeline cache management

    let active_camera = scene.create_entity();
    scene.storage::<component::Camera>().write().unwrap().set(
        active_camera,
        component::Camera::new(1.0, std::f32::consts::FRAC_PI_2, 0.01),
    );

    let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);
    let present_queue = device.get_queue(Queue::TYPE_PRESENT);
    let phys_cores = num_cpus::get_physical();

    let per_frame_uniform_buffer = device.create_device_buffer(
        BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM,
        mem::size_of::<PerFrameInfo>() as u64,
        1,
    )?;
    let material_buffer = device
        .create_device_buffer(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::STORAGE,
            mem::size_of::<MaterialInfo>() as u64,
            MAX_MATERIAL_COUNT as u64,
        )
        .unwrap();

    // Create depth pass resources
    // -----------------------------------------------------------------------------------------------------------------
    let depth_render_pass = device.create_render_pass(
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
    )?;
    let depth_vertex = device.create_shader(
        &resources.get("shaders/depth.vert.spv").unwrap().read().unwrap(),
        &[("inPosition", Format::RGB32_FLOAT)],
        &[],
    )?;
    let depth_signature = device.create_pipeline_signature(&[depth_vertex], &[])?;
    let depth_pipeline_r = device.create_graphics_pipeline(
        &depth_render_pass,
        0,
        PrimitiveTopology::TRIANGLE_LIST,
        PipelineDepthStencil::new().depth_test(true).depth_write(false),
        PipelineRasterization::new().cull_back_faces(true),
        &depth_signature,
    )?;
    let depth_pipeline_rw = device.create_graphics_pipeline(
        &depth_render_pass,
        0,
        PrimitiveTopology::TRIANGLE_LIST,
        PipelineDepthStencil::new().depth_test(true).depth_write(true),
        PipelineRasterization::new().cull_back_faces(true),
        &depth_signature,
    )?;
    let mut depth_per_frame_pool = depth_signature.create_pool(0, 1)?;
    let depth_per_frame_in = depth_per_frame_pool.alloc()?;

    let depth_secondary_cls = iter::repeat_with(|| graphics_queue.create_secondary_cmd_list().unwrap())
        .take(phys_cores)
        .collect();

    // Depth pyramid pipeline
    // -----------------------------------------------------------------------------------------------------------------

    let depth_pyramid_compute = device
        .create_shader(
            &resources
                .get("shaders/depth_pyramid.comp.spv")
                .unwrap()
                .read()
                .unwrap(),
            &[],
            &[],
        )
        .unwrap();
    let depth_pyramid_signature = device
        .create_pipeline_signature(&[depth_pyramid_compute], &[])
        .unwrap();
    let depth_pyramid_pipeline = device.create_compute_pipeline(&depth_pyramid_signature).unwrap();
    let depth_pyramid_sampler = device.create_reduction_sampler(Sampler::REDUCTION_MAX).unwrap();

    // Cull pipeline
    // -----------------------------------------------------------------------------------------------------------------
    let cull_compute = device
        .create_shader(
            &resources.get("shaders/cull.comp.spv").unwrap().read().unwrap(),
            &[],
            &[],
        )
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

    // Compose pipeline
    // -----------------------------------------------------------------------------------------------------------------
    let quad_vert_shader = device
        .create_shader(
            &resources.get("shaders/quad.vert.spv").unwrap().read().unwrap(),
            &[],
            &[],
        )
        .unwrap();
    let compose_pixel_shader = device
        .create_shader(
            &resources.get("shaders/compose.frag.spv").unwrap().read().unwrap(),
            &[("", Format::RGBA8_UNORM)],
            &[],
        )
        .unwrap();
    let compose_signature = device
        .create_pipeline_signature(&[quad_vert_shader, compose_pixel_shader], &[])
        .unwrap();
    let mut compose_pool = compose_signature.create_pool(0, 1).unwrap();
    let compose_desc = compose_pool.alloc().unwrap();

    // Create G-Buffer pass resources
    // -----------------------------------------------------------------------------------------------------------------
    let g_render_pass = device.create_render_pass(
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
                init_layout: ImageLayout::DEPTH_READ,
                final_layout: ImageLayout::DEPTH_READ,
                load_store: LoadStore::InitSave,
            },
            Attachment {
                format: Format::R32_UINT,
                init_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::GENERAL,
                load_store: LoadStore::InitClearFinalSave,
            },
        ],
        &[Subpass {
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
        }],
        &[],
    )?;
    let g_signature = device
        .create_pipeline_signature(&[], &*ADDITIONAL_PIPELINE_BINDINGS)
        .unwrap();
    let mut g_per_frame_pool = g_signature.create_pool(0, 1)?;
    let g_per_frame_in = g_per_frame_pool.alloc()?;
    let g_secondary_cls = iter::repeat_with(|| graphics_queue.create_secondary_cmd_list().unwrap())
        .take(phys_cores)
        .collect();

    let tile_count = max_texture_count;
    let texture_atlases = [
        // albedo
        texture_atlas::new(
            &device,
            Format::BC7_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            tile_count,
            settings.texture_quality as u32,
        )
        .unwrap(),
        // specular
        texture_atlas::new(
            &device,
            Format::BC7_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            tile_count,
            settings.texture_quality as u32,
        )
        .unwrap(),
        // emission
        texture_atlas::new(
            &device,
            Format::BC7_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            tile_count,
            settings.texture_quality as u32,
        )
        .unwrap(),
        // normal
        texture_atlas::new(
            &device,
            Format::BC5_RG_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            tile_count,
            settings.texture_quality as u32,
        )
        .unwrap(),
    ];

    // Update pipeline inputs
    depth_per_frame_pool.update(
        depth_per_frame_in,
        &[Binding {
            id: 0,
            array_index: 0,
            res: BindingRes::Buffer(Arc::clone(&per_frame_uniform_buffer)),
        }],
    );
    g_per_frame_pool.update(
        g_per_frame_in,
        &[
            Binding {
                id: 0,
                array_index: 0,
                res: BindingRes::Buffer(Arc::clone(&per_frame_uniform_buffer)),
            },
            Binding {
                id: 1,
                array_index: 0,
                res: BindingRes::Image(Arc::clone(&texture_atlases[0].image()), ImageLayout::SHADER_READ),
            },
            Binding {
                id: 2,
                array_index: 0,
                res: BindingRes::Image(Arc::clone(&texture_atlases[1].image()), ImageLayout::SHADER_READ),
            },
            Binding {
                id: 3,
                array_index: 0,
                res: BindingRes::Image(Arc::clone(&texture_atlases[3].image()), ImageLayout::SHADER_READ),
            },
            Binding {
                id: 4,
                array_index: 0,
                res: BindingRes::Buffer(Arc::clone(&material_buffer)),
            },
        ],
    );

    let staging_cl = graphics_queue.create_primary_cmd_list()?;
    let staging_submit =
        device.create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&staging_cl)], &[])])?;

    let final_cl = [
        graphics_queue.create_primary_cmd_list()?,
        present_queue.create_primary_cmd_list()?,
    ];
    let final_submit = [
        device.create_submit_packet(&[])?,
        device.create_submit_packet(&[])?,
    ];

    let free_indices: Vec<u32> = (0..tile_count).into_iter().collect();

    let mut renderer = Renderer {
        scene,
        surface: Arc::clone(surface),
        swapchain: None,
        surface_changed: false,
        surface_size: size,
        settings,
        device: Arc::clone(device),
        texture_resources: Default::default(),
        texture_atlases,
        free_texture_indices: [
            free_indices.clone(),
            free_indices.clone(),
            free_indices.clone(),
            free_indices.clone(),
        ],
        staging_buffer: device.create_host_buffer(BufferUsageFlags::TRANSFER_SRC, 0x800000)?,
        staging_cl,
        staging_submit: Arc::new(Mutex::new(staging_submit)),
        final_cl,
        final_submit,
        sw_framebuffers: vec![],
        visibility_buffer,
        depth_secondary_cls,
        g_secondary_cls,
        depth_render_pass,
        depth_pyramid_image: None,
        depth_framebuffer: None,
        depth_signature: Arc::clone(&depth_signature),
        depth_pipeline_r,
        depth_pipeline_rw,
        depth_per_object_pool: Some(depth_signature.create_pool(1, MAX_OBJECT_COUNT)?),
        _depth_per_frame_pool: depth_per_frame_pool,
        depth_per_frame_in,
        depth_pyramid_pipeline,
        depth_pyramid_signature,
        depth_pyramid_pool: None,
        depth_pyramid_descs: vec![],
        depth_pyramid_sampler,
        cull_pipeline,
        cull_signature,
        cull_pool,
        cull_desc: cull_descriptor,
        cull_buffer,
        g_render_pass,
        g_framebuffer: None,
        _g_per_frame_pool: g_per_frame_pool,
        g_per_frame_desc: g_per_frame_in,
        g_per_pipeline_pools: Some(Default::default()),
        translucency_head_image: None,
        translucency_texel_image: None,
        active_camera_desc: active_camera,
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
    };
    renderer.on_resize(size);

    Ok(Arc::new(Mutex::new(renderer)))
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
    }
}
