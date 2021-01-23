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

use crate::renderer::material_pipeline::{MaterialPipeline, PipelineMapping};
use crate::resource_file::{ResourceFile, ResourceRef};
use crate::utils;
use ktx::KtxInfo;
use nalgebra as na;
use nalgebra::{Matrix4, Vector4};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{mem, slice};
use texture_atlas::TextureAtlas;
use vertex_mesh::VertexMeshCmdList;
use vk_wrapper as vkw;
use vk_wrapper::queue::SignalSemaphore;
use vk_wrapper::{
    swapchain, AccessFlags, Binding, BindingRes, BindingType, DescriptorPool, HostBuffer, Image,
    ImageUsageFlags, ShaderBinding, ShaderBindingMod, ShaderStage, SwapchainImage,
};
use vk_wrapper::{
    Attachment, AttachmentRef, BufferUsageFlags, ClearValue, CmdList, Device, DeviceBuffer, Format,
    Framebuffer, ImageLayout, ImageMod, LoadStore, Pipeline, PipelineDepthStencil, PipelineRasterization,
    PipelineSignature, PipelineStageFlags, PrimitiveTopology, QueryPool, Queue, RenderPass, SubmitInfo,
    SubmitPacket, Subpass, Surface, Swapchain, WaitSemaphore,
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
    final_cl: Arc<Mutex<CmdList>>,
    final_submit: SubmitPacket,

    sw_framebuffers: Vec<Arc<Framebuffer>>,

    occlusion_buffer: HostBuffer<u32>,
    secondary_cmd_lists: Vec<Arc<Mutex<CmdList>>>,

    depth_render_pass: Arc<RenderPass>,
    depth_framebuffer: Option<Arc<Framebuffer>>,
    depth_signature: Arc<PipelineSignature>,
    depth_pipeline_r: Arc<Pipeline>,
    depth_pipeline_rw: Arc<Pipeline>,
    depth_per_object_pool: Arc<DescriptorPool>,
    depth_per_frame_pool: Arc<DescriptorPool>,
    depth_per_frame_in: u32,

    g_render_pass: Arc<RenderPass>,
    g_signature: Arc<PipelineSignature>,
    g_framebuffer: Option<Arc<Framebuffer>>,
    g_per_frame_pool: Arc<DescriptorPool>,
    g_per_frame_in: u32,
    g_per_object_pools: HashMap<Arc<Pipeline>, Arc<DescriptorPool>>,

    translucency_head_image: Option<Arc<Image>>,
    translucency_texel_image: Option<Arc<Image>>,

    model_inputs_pool: Arc<DescriptorPool>,

    active_camera: u32,
    camera_uniform_buffer: Arc<DeviceBuffer>,
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

struct BufferUpdate {
    buffer: Arc<DeviceBuffer>,
    offset: u64,
    data: Vec<u8>,
}

#[derive(Debug)]
#[repr(C)]
pub struct CameraInfo {
    pos: Vector4<f32>,
    dir: Vector4<f32>,
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    info: Vector4<f32>, // .x - FovY
}

#[repr(C)]
pub struct PerFrameInfo {
    camera: CameraInfo,
    atlas_info: na::Vector4<u32>,
}

impl Renderer {
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn get_active_camera(&self) -> u32 {
        self.active_camera
    }

    pub fn set_active_camera(&mut self, entity: u32) {
        self.active_camera = entity;
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

            let first_level = utils::log2(width / (self.settings.texture_quality as u32));
            let last_level = utils::log2(width / 4); // BC block size = 4x4

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

    pub fn prepare_material(&self, material_pipeline: &mut MaterialPipeline) {
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
    }

    /// Copy each [u8] slice to appropriate DeviceBuffer with offset u64
    fn update_device_buffers(&mut self, updates: &[BufferUpdate]) {
        if updates.is_empty() {
            return;
        }

        let update_count = updates.len();

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        self.staging_cl.lock().unwrap().begin(true).unwrap();

        let mut used_size = 0;
        let mut i = 0;

        loop {
            let update = &updates[i];
            let copy_size = update.data.len();
            let new_used_size = used_size + copy_size;

            if copy_size > 0 && new_used_size < self.staging_buffer.size() as usize {
                self.staging_buffer.write(used_size as u64, &update.data);

                let mut cl = self.staging_cl.lock().unwrap();

                cl.copy_buffer_to_device(
                    &self.staging_buffer,
                    used_size as u64,
                    &update.buffer,
                    update.offset,
                    copy_size as u64,
                );
                used_size = new_used_size;
                i += 1;
            }

            if i == update_count || new_used_size > self.staging_buffer.size() as usize {
                self.staging_cl.lock().unwrap().end().unwrap();

                let mut submit = self.staging_submit.lock().unwrap();
                graphics_queue.submit(&mut submit).unwrap();
                submit.wait().unwrap();

                if i == update_count {
                    break;
                } else if i < update_count {
                    self.staging_cl.lock().unwrap().begin(true).unwrap();
                }
            }
        }
    }

    pub fn set_settings(&mut self, settings: Settings) {
        // TODO
        self.settings = settings;
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_size = new_size;
        self.surface_changed = true;

        // Set camera aspect
        {
            let entity = self.get_active_camera();
            let camera_comps = self.scene.storage::<component::Camera>();
            let mut camera_comps = camera_comps.write().unwrap();
            let camera = camera_comps.get_mut(entity).unwrap();
            camera.set_aspect(new_size.0, new_size.1);
        }

        self.depth_framebuffer = Some(self.depth_render_pass.create_framebuffer(new_size, &[]).unwrap());

        self.g_framebuffer = Some(
            self.g_render_pass
                .create_framebuffer(
                    new_size,
                    &[
                        (
                            0,
                            ImageMod::AdditionalUsage(
                                ImageUsageFlags::INPUT_ATTACHMENT | ImageUsageFlags::TRANSFER_SRC,
                            ),
                        ),
                        (1, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                        (2, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                        (3, ImageMod::AdditionalUsage(ImageUsageFlags::INPUT_ATTACHMENT)),
                        (
                            4,
                            ImageMod::OverrideImage(self.depth_framebuffer.as_ref().unwrap().get_image(0)),
                        ),
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
                    vkw::Format::RG32_UINT, // color & depth
                    vkw::ImageUsageFlags::STORAGE,
                    (
                        new_size.0,
                        new_size.1,
                        self.settings.translucency_max_depth as u32,
                    ),
                )
                .unwrap(),
        )
    }

    pub fn on_update(&mut self) {
        let camera = {
            let camera_comps = self.scene.storage::<component::Camera>();
            let camera_comps = camera_comps.read().unwrap();
            *camera_comps.get(self.get_active_camera()).unwrap()
        };

        let mut renderer_events_system = systems::RendererCompEventsSystem {
            renderer_comps: self.scene.storage::<component::Renderer>(),
            depth_per_object_pool: Arc::clone(&self.depth_per_object_pool),
            model_inputs_pool: Arc::clone(&self.model_inputs_pool),
        };

        let mut vertex_mesh_system = systems::VertexMeshCompEventsSystem {
            vertex_mesh_comps: self.scene.storage::<component::VertexMesh>(),
            device: Arc::clone(&self.device),
            staging_cl: Arc::clone(&self.staging_cl),
            staging_submit: Arc::clone(&self.staging_submit),
        };

        let mut transform_events_system = systems::TransformEventsSystem {
            transform_comps: self.scene.storage::<component::Transform>(),
            model_transform_comps: self.scene.storage::<component::ModelTransform>(),
        };

        let buffer_updates = Arc::new(Mutex::new(vec![]));
        let mut world_transform_events_system = systems::WorldTransformEventsSystem {
            buffer_updates: Arc::clone(&buffer_updates),
            world_transform_comps: self.scene.storage::<component::WorldTransform>(),
            renderer_comps: self.scene.storage::<component::Renderer>(),
        };

        let mut hierarchy_propagation_system = systems::HierarchyPropagationSystem {
            parent_comps: self.scene.storage::<component::Parent>(),
            children_comps: self.scene.storage::<component::Children>(),
            model_transform_comps: self.scene.storage::<component::ModelTransform>(),
            world_transform_comps: self.scene.storage::<component::WorldTransform>(),
        };

        rayon::scope(|s| {
            rayon::scope(|s| {
                s.spawn(|s| {
                    renderer_events_system.run();
                });
                s.spawn(|s| {
                    vertex_mesh_system.run();
                });
                s.spawn(|s| {
                    transform_events_system.run();
                });
            });
            rayon::scope(|s| {
                hierarchy_propagation_system.run();
            });
            rayon::scope(|s| {
                s.spawn(|s| {
                    world_transform_events_system.run();
                });
            });
        });

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
                        info: Vector4::new(camera.fovy(), 0.0, 0.0, 0.0),
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
            buffer_updates.lock().unwrap().push(BufferUpdate {
                buffer: Arc::clone(&self.camera_uniform_buffer),
                offset: 0,
                data,
            });
        }

        // Wait for vertex buffer updates
        self.staging_submit.lock().unwrap().wait().unwrap();

        self.update_device_buffers(&buffer_updates.lock().unwrap());

        // let t2 = Instant::now();
        // println!("{}", (t2 - t).as_secs_f64());
    }

    fn on_render(&mut self, sw_image: &SwapchainImage) {
        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

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
        let object_count = renderables.len();
        let draw_count_step = object_count / self.secondary_cmd_lists.len() + 1;

        // Record depth object rendering
        // -------------------------------------------------------------------------------------------------------------
        self.secondary_cmd_lists
            .par_iter()
            .enumerate()
            .for_each(|(i, cmd_list)| {
                let mut cl = cmd_list.lock().unwrap();

                cl.begin_secondary_graphics(
                    true,
                    &self.depth_render_pass,
                    0,
                    Some(self.depth_framebuffer.as_ref().unwrap()),
                )
                .unwrap();

                let used_depth_pool = cl.use_descriptor_pool(Arc::clone(&self.depth_per_frame_pool));
                let used_model_inputs_pool = cl.use_descriptor_pool(Arc::clone(&self.model_inputs_pool));

                cl.bind_graphics_input(&self.depth_signature, 0, used_depth_pool, self.depth_per_frame_in);

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
                    let aabb = { *vertex_mesh.aabb() };

                    let center_position = (aabb.0 + aabb.1) * 0.5 + transform.position;
                    let radius = ((aabb.1 - aabb.0).component_mul(&transform.scale) * 0.5).magnitude();

                    // TODO ------------------------------------------
                    // TODO: -> REIMPLEMENT: 1. firstly render depth buffer without queries; 2. render object AABBs using queries
                    // TODO: Many queries is slow approach, use max 128 queries (distribute one query across multiple objects)
                    // TODO: OR implement compute-based approach (https://vkguide.dev/docs/gpudriven/compute_culling)
                    // TODO ------------------------------------------

                    if !camera.is_sphere_visible(center_position, radius) || vertex_mesh.vertex_count == 0 {
                        continue;
                    }

                    if renderer.translucent {
                        cl.bind_pipeline(&self.depth_pipeline_r);
                    } else {
                        cl.bind_pipeline(&self.depth_pipeline_rw);
                    }

                    cl.bind_graphics_input(
                        &self.depth_signature,
                        1,
                        used_model_inputs_pool,
                        renderer.pipeline_inputs[0],
                    );
                    cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                }

                cl.end().unwrap();
            });

        // Record depth cmd list
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut cl = self.staging_cl.lock().unwrap();
            cl.begin(true).unwrap();
            cl.begin_render_pass(
                &self.depth_render_pass,
                self.depth_framebuffer.as_ref().unwrap(),
                &[ClearValue::Depth(1.0)],
                true,
            );
            cl.execute_secondary(&self.secondary_cmd_lists);
            cl.end_render_pass();
            cl.end().unwrap();
        }
        {
            let mut submit = self.staging_submit.lock().unwrap();
            graphics_queue.submit(&mut submit).unwrap();
            submit.wait().unwrap();
        }

        // TODO: separate depth & g-buffer cmd lists (for performance reasons)

        // Record g-buffer object rendering
        // -------------------------------------------------------------------------------------------------------------
        let pipeline_mapping = PipelineMapping {
            render_pass: Arc::clone(&self.g_render_pass),
            subpass_index: 0,
            cull_back_faces: true,
        };
        let t0 = Instant::now();
        self.secondary_cmd_lists
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

                let used_g_per_frame_pool = cl.use_descriptor_pool(Arc::clone(&self.g_per_frame_pool));
                let used_model_inputs_pool = cl.use_descriptor_pool(Arc::clone(&self.model_inputs_pool));

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
                    if self.occlusion_buffer[entity_index] == 0 {
                        // TODO
                        // continue;
                    }

                    let mesh = vertex_mesh.unwrap();
                    let vertex_mesh = &mesh.0;

                    let mat_pipeline = &renderer.mat_pipeline;
                    let pipeline = mat_pipeline.get_pipeline(&pipeline_mapping).unwrap();
                    let signature = pipeline.signature();

                    let already_bound = cl.bind_pipeline(pipeline);
                    if !already_bound {
                        cl.bind_graphics_input(&signature, 0, used_g_per_frame_pool, self.g_per_frame_in);
                    }
                    cl.bind_graphics_input(
                        &signature,
                        1,
                        used_model_inputs_pool,
                        renderer.pipeline_inputs[1],
                    );
                    cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                }

                cl.end().unwrap();
            });
        let t1 = Instant::now();
        println!("g rec {}", (t1 - t0).as_secs_f64());

        // Record G-Buffer cmd list
        // -------------------------------------------------------------------------------------------------------------
        {
            // !!! Do not render anything in final cl except copying some image into swapchain image.
            // Uniform/vertex buffers may be being updated at this moment.
            let mut cl = self.final_cl.lock().unwrap();
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
            cl.execute_secondary(&self.secondary_cmd_lists);
            cl.end_render_pass();

            let sw_image = self.sw_framebuffers[sw_image.get_index() as usize].get_image(0);
            let albedo = self.g_framebuffer.as_ref().unwrap().get_image(0);

            cl.barrier_image(
                PipelineStageFlags::BOTTOM_OF_PIPE,
                PipelineStageFlags::TRANSFER,
                &[
                    albedo.barrier_queue(
                        AccessFlags::COLOR_ATTACHMENT_WRITE,
                        AccessFlags::TRANSFER_READ,
                        ImageLayout::SHADER_READ,
                        ImageLayout::TRANSFER_SRC,
                        graphics_queue,
                        graphics_queue,
                    ),
                    sw_image.barrier_queue(
                        AccessFlags::default(),
                        AccessFlags::TRANSFER_WRITE,
                        ImageLayout::UNDEFINED,
                        ImageLayout::TRANSFER_DST,
                        graphics_queue,
                        graphics_queue,
                    ),
                ],
            );
            cl.blit_image_2d(
                &albedo,
                ImageLayout::TRANSFER_SRC,
                (0, 0),
                albedo.size_2d(),
                0,
                &sw_image,
                ImageLayout::TRANSFER_DST,
                (0, 0),
                sw_image.size_2d(),
                0,
            );
            cl.barrier_image(
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::BOTTOM_OF_PIPE,
                &[
                    albedo.barrier_queue(
                        AccessFlags::TRANSFER_READ,
                        AccessFlags::default(),
                        ImageLayout::TRANSFER_SRC,
                        ImageLayout::SHADER_READ,
                        graphics_queue,
                        graphics_queue,
                    ),
                    sw_image.barrier_queue(
                        AccessFlags::TRANSFER_WRITE,
                        AccessFlags::default(),
                        ImageLayout::TRANSFER_DST,
                        ImageLayout::PRESENT,
                        graphics_queue,
                        graphics_queue,
                    ),
                ],
            );
            cl.end().unwrap();
        }

        graphics_queue.submit(&mut self.final_submit).unwrap();
    }

    pub fn on_draw(&mut self) {
        let device = Arc::clone(&self.device);
        let adapter = device.get_adapter();
        let surface = &self.surface;

        if adapter.is_surface_valid(surface).unwrap() {
            if self.surface_changed {
                let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);

                self.swapchain = Some(
                    device
                        .create_swapchain(&self.surface, self.surface_size, self.settings.vsync)
                        .unwrap(),
                );
                self.final_submit
                    .set(&[SubmitInfo::new(
                        &[WaitSemaphore {
                            semaphore: Arc::clone(self.swapchain.as_ref().unwrap().get_semaphore()),
                            wait_dst_mask: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, // TODO: change if necessary
                            wait_value: 0,
                        }],
                        &[Arc::clone(&self.final_cl)],
                        &[SignalSemaphore {
                            semaphore: Arc::clone(present_queue.frame_semaphore()),
                            signal_value: 0,
                        }],
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

                    self.on_update();
                    self.final_submit.wait().unwrap();
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

    fn create_main_framebuffers(&mut self) {
        self.sw_framebuffers.clear();

        let images = self.swapchain.as_ref().unwrap().get_images();

        let sw_render_pass = self
            .device
            .create_render_pass(
                &[Attachment {
                    format: images[0].format(),
                    init_layout: ImageLayout::UNDEFINED,
                    final_layout: ImageLayout::PRESENT,
                    load_store: LoadStore::InitClearFinalSave,
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
            .unwrap();

        for img in images {
            self.sw_framebuffers.push(
                sw_render_pass
                    .create_framebuffer(
                        images[0].size_2d(),
                        &[(0, ImageMod::OverrideImage(Arc::clone(img)))],
                    )
                    .unwrap(),
            );
        }
    }
}

pub fn new(
    surface: &Arc<Surface>,
    size: (u32, u32),
    settings: Settings,
    device: &Arc<Device>,
    resources: &Arc<ResourceFile>,
    max_texture_count: u32,
) -> Result<Arc<Mutex<Renderer>>, vkw::DeviceError> {
    let mut scene = Scene::new();

    // TODO: pipeline cache management

    let active_camera = scene.create_entity();
    scene.storage::<component::Camera>().write().unwrap().set(
        active_camera,
        component::Camera::new(1.0, std::f32::consts::FRAC_PI_2, 0.01),
    );

    let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);

    // Create secondary cmd lists for multithread recording
    let mut secondary_cmd_lists = vec![];
    for _ in 0..num_cpus::get_physical().min(8) {
        secondary_cmd_lists.push(graphics_queue.create_secondary_cmd_list()?);
    }

    // Create per-frame uniform buffer
    let per_frame_uniform_buffer = device.create_device_buffer(
        BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM,
        mem::size_of::<PerFrameInfo>() as u64,
        1,
    )?;

    // Create depth pass resources
    // -----------------------------------------------------------------------------------------------------------------
    let depth_render_pass = device.create_render_pass(
        &[Attachment {
            format: Format::D32_FLOAT,
            init_layout: ImageLayout::UNDEFINED,
            final_layout: ImageLayout::DEPTH_READ,
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
    let depth_signature = device.create_pipeline_signature(&[depth_vertex])?;
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
    let depth_per_frame_pool = depth_signature.create_pool(0, 1)?;
    let depth_per_frame_in = depth_per_frame_pool.lock().unwrap().alloc()?;

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
        .create_custom_pipeline_signature(&[
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
                ],
            ),
        ])
        .unwrap();
    let g_per_frame_pool = g_signature.create_pool(0, 1)?;
    let g_per_frame_in = g_per_frame_pool.lock().unwrap().alloc()?;

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
    depth_per_frame_pool.lock().unwrap().update(
        depth_per_frame_in,
        &[Binding {
            id: 0,
            array_index: 0,
            res: BindingRes::Buffer(Arc::clone(&per_frame_uniform_buffer)),
        }],
    );
    g_per_frame_pool.lock().unwrap().update(
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
        ],
    );

    let staging_cl = graphics_queue.create_primary_cmd_list()?;
    let staging_submit =
        device.create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&staging_cl)], &[])])?;

    let final_cl = graphics_queue.create_primary_cmd_list()?;
    let final_submit = device.create_submit_packet(&[])?;

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
        occlusion_buffer: device.create_host_buffer(
            vkw::BufferUsageFlags::TRANSFER_DST,
            (mem::size_of::<u32>() * 65535) as u64,
        )?,
        secondary_cmd_lists,
        depth_render_pass,
        depth_framebuffer: None,
        depth_signature: Arc::clone(&depth_signature),
        depth_pipeline_r,
        depth_pipeline_rw,
        depth_per_frame_pool,
        depth_per_frame_in,
        depth_per_object_pool: depth_signature.create_pool(1, 65535)?,
        g_render_pass,
        g_signature: Arc::clone(&g_signature),
        g_framebuffer: None,
        g_per_frame_pool,
        g_per_frame_in,
        g_per_object_pools: Default::default(),
        translucency_head_image: None,
        translucency_texel_image: None,
        model_inputs_pool: g_signature.create_pool(1, 65535)?, // TODO: REMOVE
        active_camera,
        camera_uniform_buffer: per_frame_uniform_buffer,
    };
    renderer.on_resize(size);

    Ok(Arc::new(Mutex::new(renderer)))
}
