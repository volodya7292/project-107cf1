pub(crate) mod component;
#[macro_use]
pub(crate) mod material_pipeline;
pub mod material_pipelines;
mod texture_atlas;
#[macro_use]
pub(crate) mod vertex_mesh;

use crate::renderer::texture_atlas::TextureAtlas;
use crate::resource_file::{ResourceFile, ResourceRef};
use crate::utils;
use ktx::KtxInfo;
use nalgebra as na;
use nalgebra::{Matrix4, Vector4};
use order_stat;
use rayon::prelude::*;
use specs::prelude::ParallelIterator;
use specs::storage::ComponentEvent;
use specs::WorldExt;
use specs::{Builder, Join, ParJoin};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::{cmp, mem, slice};
use vertex_mesh::VertexMeshCmdList;
use vk_wrapper as vkw;
use vk_wrapper::pipeline_input::{Binding, BindingRes};
use vk_wrapper::{
    swapchain, AccessFlags, BindingType, DescriptorPool, HostBuffer, Image, ImageUsageFlags, ShaderBinding,
    ShaderBindingMod, ShaderStage, SwapchainImage,
};
use vk_wrapper::{
    Attachment, AttachmentRef, BufferUsageFlags, ClearValue, CmdList, Device, DeviceBuffer, Format,
    Framebuffer, ImageLayout, ImageMod, LoadStore, Pipeline, PipelineDepthStencil, PipelineInput,
    PipelineRasterization, PipelineSignature, PipelineStageFlags, PrimitiveTopology, QueryPool, Queue,
    RenderPass, SubmitInfo, SubmitPacket, Subpass, Surface, Swapchain, WaitSemaphore,
};

const DISTANCE_SORT_PER_UPDATE: u32 = 128;

pub struct Renderer {
    world: specs::World,
    renderer_cmp_reader: specs::ReaderId<specs::storage::ComponentEvent>,
    sorted_render_entities: Vec<specs::Entity>,
    sort_count: u32,

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
    staging_submit: SubmitPacket,

    sw_framebuffers: Vec<Arc<Framebuffer>>,

    query_pool: Arc<QueryPool>,
    occlusion_buffer: HostBuffer<u32>,
    secondary_cmd_lists: Vec<Arc<Mutex<CmdList>>>,

    depth_render_pass: Arc<RenderPass>,
    depth_framebuffer: Option<Arc<Framebuffer>>,
    depth_signature: Arc<PipelineSignature>,
    depth_pipeline_r: Arc<Pipeline>,
    depth_pipeline_rw: Arc<Pipeline>,
    depth_per_frame_in: Arc<PipelineInput>,
    depth_per_object_pool: Arc<DescriptorPool>,

    g_render_pass: Arc<RenderPass>,
    g_signature: Arc<PipelineSignature>,
    g_framebuffer: Option<Arc<Framebuffer>>,
    g_per_frame_in: Arc<PipelineInput>,
    g_per_object_pools: HashMap<Arc<Pipeline>, Arc<DescriptorPool>>,

    translucency_head_image: Option<Arc<Image>>,
    translucency_texel_image: Option<Arc<Image>>,

    model_inputs: Arc<DescriptorPool>,

    active_camera: specs::Entity,
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

#[derive(Debug)]
pub struct CameraInfo {
    pos: Vector4<f32>,
    dir: Vector4<f32>,
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    info: Vector4<f32>, // .x - FovY
}

pub struct PerFrameInfo {
    camera: CameraInfo,
    atlas_info: na::Vector4<u32>,
}

/*struct RenderSystem;

impl<'a> specs::System<'a> for RenderSystem {
    type SystemData = (
        specs::ReadStorage<'a, component::Transform>,
        specs::ReadStorage<'a, component::Renderer>,
    );

    fn run(&mut self, (transform, renderer): Self::SystemData) {
        for (trans, rend) in (&transform, &renderer).join() {}
    }
}
*/

impl Renderer {
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn world(&self) -> &specs::World {
        &self.world
    }

    pub fn world_mut(&mut self) -> &mut specs::World {
        &mut self.world
    }

    pub fn add_entity(&mut self) -> specs::EntityBuilder {
        self.world.create_entity()
    }

    pub fn get_active_camera(&self) -> specs::Entity {
        self.active_camera
    }

    pub fn set_active_camera(&mut self, entity: specs::Entity) {
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

    /// Copy each [u8] slice to appropriate DeviceBuffer with offset u64
    fn update_device_buffers(&mut self, updates: &[(Vec<u8>, Arc<DeviceBuffer>, u64)]) {
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
            let copy_size = update.0.len();
            let new_used_size = used_size + copy_size;

            if copy_size > 0 && new_used_size < self.staging_buffer.size() as usize {
                self.staging_buffer.write(used_size as u64, &update.0);

                let mut cl = self.staging_cl.lock().unwrap();

                cl.copy_buffer_to_device(
                    &self.staging_buffer,
                    used_size as u64,
                    &update.1,
                    update.2,
                    copy_size as u64,
                );
                used_size = new_used_size;
                i += 1;
            }

            if i == update_count || new_used_size > self.staging_buffer.size() as usize {
                self.staging_cl.lock().unwrap().end().unwrap();

                graphics_queue.submit(&mut self.staging_submit).unwrap();
                self.staging_submit.wait().unwrap();

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
            let mut camera_comp = self.world.write_component::<component::Camera>();
            let camera = camera_comp.get_mut(entity).unwrap();
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
        let mut removed_entity_count = 0;

        // Add new objects to sort
        // -------------------------------------------------------------------------------------------------------------
        {
            let entities = self.world.entities();
            let renderer_comp = self.world.read_component::<component::Renderer>();
            let renderer_comp_events = renderer_comp.channel().read(&mut self.renderer_cmp_reader);

            let mut inserted = specs::BitSet::new();

            for event in renderer_comp_events {
                match event {
                    ComponentEvent::Inserted(i) => {
                        inserted.add(*i);
                    }
                    ComponentEvent::Removed(_) => {
                        removed_entity_count += 1;
                    }
                    _ => {}
                }
            }

            for (entity, _comp, _) in (&entities, &renderer_comp, &inserted).join() {
                self.sorted_render_entities.push(entity);
            }
        }

        self.world.maintain();

        // Replace removed(dead) entities with alive ones
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut swap_entities = Vec::<specs::Entity>::with_capacity(removed_entity_count);
            let mut new_len = self.sorted_render_entities.len();

            // Find alive entities for replacement
            for &entity in self.sorted_render_entities.iter().rev() {
                if self.world.is_alive(entity) {
                    if removed_entity_count > swap_entities.len() {
                        swap_entities.push(entity);
                    } else {
                        break;
                    }
                }
                new_len -= 1;
            }

            // Resize vector to trim swapped entities
            if !self.sorted_render_entities.is_empty() {
                let def_entity = self.sorted_render_entities[0];
                self.sorted_render_entities.resize(new_len, def_entity);
            }

            // Swap entities
            for entity in &mut self.sorted_render_entities {
                if !self.world.is_alive(*entity) {
                    *entity = swap_entities.remove(swap_entities.len() - 1);
                }
            }

            // Add the rest of swap_entities that were not swapped due to resized vector
            self.sorted_render_entities.extend(swap_entities);
        }

        // Sort render objects from front to back (for Z rejection & occlusion queries)
        // -------------------------------------------------------------------------------------------------------------
        {
            let camera_comp = self.world.read_component::<component::Camera>();
            let transform_comp = self.world.read_component::<component::Transform>();
            let mesh_ref_comp = self.world.read_component::<component::VertexMeshRef>();

            let camera = camera_comp.get(self.active_camera).unwrap();
            let camera_pos = camera.position().clone();

            let sort_slice = &mut self.sorted_render_entities[(self.sort_count as usize)..];
            let to_sort_count = sort_slice.len().min(DISTANCE_SORT_PER_UPDATE as usize);

            if to_sort_count > 0 {
                order_stat::kth_by(sort_slice, to_sort_count - 1, |&a, &b| {
                    let a_transform = transform_comp.get(a);
                    let a_mesh_ref = mesh_ref_comp.get(a);
                    let b_transform = transform_comp.get(b);
                    let b_mesh_ref = mesh_ref_comp.get(b);

                    if a_transform.is_none()
                        || a_mesh_ref.is_none()
                        || b_transform.is_none()
                        || b_mesh_ref.is_none()
                    {
                        return cmp::Ordering::Equal;
                    }

                    let a_transform = a_transform.unwrap();
                    let a_mesh_ref = a_mesh_ref.unwrap();
                    let b_transform = b_transform.unwrap();
                    let b_mesh_ref = b_mesh_ref.unwrap();

                    let a_pos = {
                        let vertex_mesh = &a_mesh_ref.vertex_mesh;
                        let aabb = {
                            let vertex_mesh = vertex_mesh.lock().unwrap();
                            *vertex_mesh.aabb()
                        };
                        (aabb.0 + aabb.1) * 0.5 + a_transform.position()
                    };
                    let b_pos = {
                        let vertex_mesh = &b_mesh_ref.vertex_mesh;
                        let aabb = {
                            let vertex_mesh = vertex_mesh.lock().unwrap();
                            *vertex_mesh.aabb()
                        };
                        (aabb.0 + aabb.1) * 0.5 + b_transform.position()
                    };

                    let a_dist = (a_pos - camera_pos).magnitude();
                    let b_dist = (b_pos - camera_pos).magnitude();

                    if a_dist < b_dist {
                        cmp::Ordering::Less
                    } else if a_dist > b_dist {
                        cmp::Ordering::Greater
                    } else {
                        cmp::Ordering::Equal
                    }
                });
            }

            self.sort_count += to_sort_count as u32;
            if self.sort_count >= self.sorted_render_entities.len() as u32 {
                self.sort_count = 0;
            }
        }

        // Update pipeline inputs
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut renderer_comp = self.world.write_component::<component::Renderer>();

            (&mut renderer_comp.par_restrict_mut())
                .par_join()
                .for_each(|mut comps| {
                    let renderer = comps.get_unchecked();

                    if renderer.changed {
                        let renderer = comps.get_mut_unchecked();
                        let _mat_pipeline = &renderer.mat_pipeline;
                        let inputs = &mut renderer.pipeline_inputs;

                        inputs.clear();

                        let depth_per_object = self.depth_per_object_pool.allocate_input().unwrap();
                        depth_per_object.update(&[Binding {
                            id: 0,
                            array_index: 0,
                            res: BindingRes::Buffer(Arc::clone(&renderer.uniform_buffer)),
                        }]);

                        //let g_per_object = self.g_

                        let uniform_input = self.model_inputs.allocate_input().unwrap();
                        uniform_input.update(&[Binding {
                            id: 0,
                            array_index: 0,
                            res: BindingRes::Buffer(Arc::clone(&renderer.uniform_buffer)),
                        }]);

                        inputs.extend_from_slice(&[depth_per_object, uniform_input]);

                        renderer.changed = false;
                    }
                });
        }

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

        // Update device buffers of vertex meshes
        // -------------------------------------------------------------------------------------------------------------
        {
            let mut mesh_ref_comp = self.world.read_component::<component::VertexMeshRef>();

            self.staging_cl.lock().unwrap().begin(true).unwrap();

            (&mut mesh_ref_comp).par_join().for_each(|mesh_ref| {
                let mut vertex_mesh = mesh_ref.vertex_mesh.lock().unwrap();

                if vertex_mesh.changed {
                    let mut cl = self.staging_cl.lock().unwrap();
                    cl.copy_buffer_to_device(
                        vertex_mesh.staging_buffer.as_ref().unwrap(),
                        0,
                        vertex_mesh.buffer.as_ref().unwrap(),
                        0,
                        vertex_mesh.staging_buffer.as_ref().unwrap().size(),
                    );

                    vertex_mesh.changed = false;
                }
            });

            self.staging_cl.lock().unwrap().end().unwrap();

            graphics_queue.submit(&mut self.staging_submit).unwrap();
            // TODO: Make more efficient
            self.staging_submit.wait().unwrap();
        }

        // Check for transform updates
        // -------------------------------------------------------------------------------------------------------------
        let buffer_updates = Mutex::new(vec![]);
        {
            let mut transform_comp = self.world.write_component::<component::Transform>();
            let renderer_comp = self.world.read_component::<component::Renderer>();

            (&mut transform_comp, &renderer_comp)
                .par_join()
                .for_each(|(transform, renderer)| {
                    if transform.changed {
                        let matrix = transform.matrix();
                        let matrix_bytes = unsafe {
                            slice::from_raw_parts(
                                &matrix as *const na::Matrix4<f32> as *const u8,
                                mem::size_of::<na::Matrix4<f32>>(),
                            )
                            .to_vec()
                        };

                        buffer_updates.lock().unwrap().push((
                            matrix_bytes,
                            Arc::clone(&renderer.uniform_buffer),
                            renderer.mat_pipeline.uniform_buffer_offset_model() as u64,
                        ));

                        transform.changed = false;
                    }
                });
        }

        self.update_device_buffers(buffer_updates.lock().unwrap().as_slice());

        // Update camera uniform buffers
        // -------------------------------------------------------------------------------------------------------------
        {
            let per_frame_info = {
                let camera_comp = self.world.read_component::<component::Camera>();
                let camera = camera_comp.get(self.active_camera).unwrap();

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

            self.update_device_buffers(&[(
                unsafe {
                    slice::from_raw_parts(
                        &per_frame_info as *const PerFrameInfo as *const u8,
                        mem::size_of_val(&per_frame_info),
                    )
                    .to_vec()
                },
                Arc::clone(&self.camera_uniform_buffer),
                0,
            )]);
        }
    }

    fn on_render(&mut self, sw_image: &SwapchainImage) -> u64 {
        let camera_component = self.world.read_component::<component::Camera>();
        let active_camera = camera_component.get(self.active_camera).unwrap();

        let transform_comp = self.world.read_component::<component::Transform>();
        let renderer_comp = self.world.read_component::<component::Renderer>();
        let mesh_ref_comp = self.world.read_component::<component::VertexMeshRef>();

        let object_count = self.sorted_render_entities.len();
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

                cl.bind_graphics_input(&self.depth_signature, 0, &self.depth_per_frame_in);

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let entity = self.sorted_render_entities[entity_index];

                    let transform = transform_comp.get(entity);
                    let renderer = renderer_comp.get(entity);
                    let mesh_ref = mesh_ref_comp.get(entity);

                    if transform.is_none() || renderer.is_none() || mesh_ref.is_none() {
                        continue;
                    }

                    let transform = transform.unwrap();
                    let renderer = renderer.unwrap();
                    let mesh_ref = mesh_ref.unwrap();

                    let vertex_mesh = &mesh_ref.vertex_mesh;
                    let aabb = {
                        let vertex_mesh = vertex_mesh.lock().unwrap();
                        *vertex_mesh.aabb()
                    };

                    let center_position = (aabb.0 + aabb.1) * 0.5 + transform.position();
                    let radius = ((aabb.1 - aabb.0).component_mul(transform.scale()) * 0.5).magnitude();

                    cl.begin_query(&self.query_pool, entity_index as u32);

                    if active_camera.is_sphere_visible(center_position, radius)
                        && vertex_mesh.lock().unwrap().vertex_count > 0
                    {
                        if renderer.translucent {
                            cl.bind_pipeline(&self.depth_pipeline_r);
                        } else {
                            cl.bind_pipeline(&self.depth_pipeline_rw);
                        }

                        cl.bind_graphics_input(&self.depth_signature, 1, &renderer.pipeline_inputs[0]);
                        cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                    }

                    cl.end_query(entity_index as u32);
                }

                cl.end().unwrap();
            });

        // Record depth cmd list
        // -------------------------------------------------------------------------------------------------------------
        {
            let object_count = self.sorted_render_entities.len() as u32;

            let mut cl = self.staging_cl.lock().unwrap();
            cl.begin(true).unwrap();
            cl.reset_query_pool(&self.query_pool, 0, object_count);
            cl.begin_render_pass(
                &self.depth_render_pass,
                self.depth_framebuffer.as_ref().unwrap(),
                &[ClearValue::Depth(1.0)],
                true,
            );
            cl.execute_secondary(&self.secondary_cmd_lists);
            cl.end_render_pass();
            cl.copy_query_pool_results_to_host(&self.query_pool, 0, object_count, &self.occlusion_buffer, 0);
            cl.end().unwrap();
        }

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

        graphics_queue.submit(&mut self.staging_submit).unwrap();
        self.staging_submit.wait().unwrap();

        // Record g-buffer object rendering
        // -------------------------------------------------------------------------------------------------------------
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

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let entity = self.sorted_render_entities[entity_index];

                    let renderer = renderer_comp.get(entity);
                    let mesh_ref = mesh_ref_comp.get(entity);

                    if renderer.is_none() || mesh_ref.is_none() {
                        continue;
                    }

                    // Check query_pool occlusion results
                    if self.occlusion_buffer[entity_index] == 0 {
                        continue;
                    }

                    let renderer = renderer.unwrap();
                    let mesh_ref = mesh_ref.unwrap();

                    let vertex_mesh = &mesh_ref.vertex_mesh;

                    let mat_pipeline = &renderer.mat_pipeline;
                    let pipeline = mat_pipeline.request_pipeline(&self.g_render_pass, 0, false);
                    let signature = pipeline.signature();

                    let already_bound = cl.bind_pipeline(&pipeline);
                    if !already_bound {
                        cl.bind_graphics_input(&signature, 0, &self.g_per_frame_in);
                    }
                    cl.bind_graphics_input(&signature, 1, &renderer.pipeline_inputs[1]);
                    cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                }

                cl.end().unwrap();
            });

        // Record G-Buffer cmd list
        // -------------------------------------------------------------------------------------------------------------
        {
            let _translucency_head_image = self.translucency_head_image.as_ref().unwrap();

            let mut cl = self.staging_cl.lock().unwrap();
            cl.begin(true).unwrap();

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

            let albedo = self.g_framebuffer.as_ref().unwrap().get_image(0);
            let sw_image = self.sw_framebuffers[sw_image.get_index() as usize].get_image(0);

            cl.barrier_image(
                PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
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

        let mut final_packet = self
            .device
            .create_submit_packet(&[SubmitInfo::new(
                &[WaitSemaphore {
                    semaphore: self.swapchain.as_ref().unwrap().get_semaphore(),
                    wait_dst_mask: PipelineStageFlags::TOP_OF_PIPE,
                    wait_value: 0,
                }],
                &[Arc::clone(&self.staging_cl)],
            )])
            .unwrap();
        graphics_queue.submit(&mut final_packet).unwrap();

        final_packet.get_signal_value(0).unwrap()
    }

    pub fn on_draw(&mut self) {
        let device = Arc::clone(&self.device);
        let adapter = device.get_adapter();
        let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);
        let surface = &self.surface;

        if adapter.is_surface_valid(surface).unwrap() {
            if self.surface_changed {
                self.swapchain = Some(
                    device
                        .create_swapchain(&self.surface, self.surface_size, self.settings.vsync)
                        .unwrap(),
                );
                self.create_main_framebuffers();
                self.surface_changed = false;
            }

            let swapchain = Arc::clone(self.swapchain.as_ref().unwrap());
            let acquire_result = swapchain.acquire_image();

            if let Ok((sw_image, optimal)) = acquire_result {
                if !optimal {
                    self.surface_changed = true;
                }

                self.on_update();
                let wait_value = self.on_render(&sw_image);

                let present_queue = self.device.get_queue(Queue::TYPE_PRESENT);
                if !present_queue
                    .present(sw_image, &graphics_queue.get_semaphore(), wait_value)
                    .unwrap_or(false)
                {
                    self.surface_changed = true;
                }
            } else {
                match acquire_result {
                    Err(swapchain::Error::IncompatibleSurface) => {
                        self.surface_changed = true;
                    }
                    _ => {
                        acquire_result.unwrap();
                    }
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
    let mut world = specs::World::new();
    world.register::<component::Transform>();
    world.register::<component::VertexMeshRef>();
    world.register::<component::Renderer>();
    world.register::<component::Camera>();

    // Register reader for listening for creation/removing of component::Renderer.
    // Used to optimize front-to-back distance sorting.
    let renderer_cmp_reader = world.write_component::<component::Renderer>().register_reader();

    // TODO: pipeline cache management

    let active_camera = world
        .create_entity()
        .with(component::Camera::new(1.0, std::f32::consts::FRAC_PI_2, 0.01))
        .build();

    let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);

    // Create secondary cmd lists for multithread recording
    let mut secondary_cmd_lists = vec![];
    for _ in 0..num_cpus::get() {
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
                layout: ImageLayout::DEPTH_ATTACHMENT,
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
    let depth_per_frame_in = depth_signature.create_pool(0, 1)?.allocate_input()?;

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
                load_store: LoadStore::InitSaveFinalSave,
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
                layout: ImageLayout::DEPTH_ATTACHMENT,
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
    let g_per_frame_in = g_signature.create_pool(0, 1)?.allocate_input().unwrap();

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
    depth_per_frame_in.update(&[Binding {
        id: 0,
        array_index: 0,
        res: BindingRes::Buffer(Arc::clone(&per_frame_uniform_buffer)),
    }]);
    g_per_frame_in.update(&[
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
    ]);

    let staging_copy_cl = graphics_queue.create_primary_cmd_list()?;
    let staging_copy_submit =
        device.create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&staging_copy_cl)])])?;

    let free_indices: Vec<u32> = (0..tile_count).into_iter().collect();

    let mut renderer = Renderer {
        world,
        renderer_cmp_reader,
        sorted_render_entities: vec![],
        sort_count: 0,
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
        staging_cl: staging_copy_cl,
        staging_submit: staging_copy_submit,
        sw_framebuffers: vec![],
        query_pool: device.create_query_pool(65536)?,
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
        depth_per_frame_in,
        depth_per_object_pool: depth_signature.create_pool(1, 65535)?,
        g_render_pass,
        g_signature: Arc::clone(&g_signature),
        g_framebuffer: None,
        g_per_frame_in,
        g_per_object_pools: Default::default(),
        translucency_head_image: None,
        translucency_texel_image: None,
        model_inputs: g_signature.create_pool(1, 65535)?, // TODO: REMOVE
        active_camera,
        camera_uniform_buffer: per_frame_uniform_buffer,
    };
    renderer.on_resize(size);

    Ok(Arc::new(Mutex::new(renderer)))
}
