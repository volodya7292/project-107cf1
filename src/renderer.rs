pub(crate) mod component;
pub(crate) mod material_pipeline;
//mod texture;
mod texture_atlas;
#[macro_use]
pub(crate) mod vertex_mesh;

use crate::renderer::texture_atlas::TextureAtlas;
use crate::resource_file::{ResourceFile, ResourceRef};
use image::GenericImageView;
use nalgebra as na;
use nalgebra::{Matrix4, Vector3, Vector4};
use order_stat;
use rayon::prelude::*;
use specs::prelude::ParallelIterator;
use specs::storage::ComponentEvent;
use specs::WorldExt;
use specs::{Builder, Join};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::{cmp, mem, slice};
use vertex_mesh::VertexMeshCmdList;
use vk_wrapper as vkw;
use vk_wrapper::{swapchain, HostBuffer, SwapchainImage};
use vk_wrapper::{
    Attachment, AttachmentRef, BufferUsageFlags, ClearValue, CmdList, Device, DeviceBuffer, Format,
    Framebuffer, ImageLayout, ImageMod, LoadStore, Pipeline, PipelineDepthStencil, PipelineInput,
    PipelineRasterization, PipelineSignature, PipelineStageFlags, PrimitiveTopology, QueryPool, Queue,
    QueueType, RenderPass, SubmitInfo, SubmitPacket, Subpass, Surface, Swapchain, WaitSemaphore,
    FORMAT_SIZES,
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

    texture_atlases: [TextureAtlas; 4],

    //submit_packet: Option<SubmitPacket>,
    sig: Arc<PipelineSignature>,
    pipe: Option<Arc<Pipeline>>,
    rp: Option<Arc<RenderPass>>,
    vb: Arc<DeviceBuffer>,

    staging_buffer: HostBuffer<u8>,
    staging_cmd_lists: Vec<Arc<Mutex<CmdList>>>,
    staging_copy_submit: SubmitPacket,

    sw_framebuffers: Vec<Arc<Framebuffer>>,

    query_pool: Arc<QueryPool>,
    secondary_cmd_lists: Vec<Arc<Mutex<CmdList>>>,

    depth_render_pass: Arc<RenderPass>,
    depth_framebuffer: Option<Arc<Framebuffer>>,
    depth_pipeline_r: Arc<Pipeline>,
    depth_pipeline_rw: Arc<Pipeline>,
    depth_camera_input: Arc<PipelineInput>,

    active_camera: specs::Entity,
    camera_uniform_buffer: Arc<DeviceBuffer>,
}

#[derive(Copy, Clone)]
pub enum TextureQuality {
    LOW,
    MEDIUM,
    HIGH,
}

#[derive(Copy, Clone)]
pub struct Settings {
    pub(crate) vsync: bool,
    pub(crate) texture_quality: TextureQuality,
    pub(crate) textures_gen_mipmaps: bool,
    pub(crate) textures_max_anisotropy: f32,
}

pub struct TextureAtlasType(u32);

impl TextureAtlasType {
    pub const ALBEDO: Self = Self(0);
    pub const SPECULAR: Self = Self(1);
    pub const EMISSION: Self = Self(2);
    pub const NORMAL: Self = Self(3);
}

pub struct CameraInfo {
    pos: Vector4<f32>,
    dir: Vector4<f32>,
    proj: Matrix4<f32>,
    view: Matrix4<f32>,
    proj_view: Matrix4<f32>,
    //vec4 info;// .x - FovY
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
    pub fn add_entity(&mut self) -> specs::EntityBuilder {
        self.world.create_entity()
    }

    pub fn set_active_camera(&mut self, entity: specs::Entity) {
        self.active_camera = entity;
    }

    /// Add texture to renderer
    pub fn add_texture(&mut self, res_ref: ResourceRef, atlas_type: TextureAtlasType) -> u16 {
        /*self.textures.push(Texture {
            res_ref,
            bounds: (0, 0, 0, 0),
            loaded: false,
            atlas_type,
        });
        (self.textures.len() - 1) as u16*/
        0
    }

    /// Texture must be loaded before use in a shader
    pub fn load_texture(&mut self, index: u16) {
        // TODO: generate mipmaps if needed

        // 1. load texture
        // 2. find space for it
        // 3. defragment atlas if needed (make some textures smaller)
        // 4. copy texture to atlas

        /*if let Some(texture) = self.textures.get_mut(index as usize) {
            // Load resource
            let bytes = texture.res_ref.read().unwrap();
            let img = image::load_from_memory(&bytes).unwrap();

            // Create staging buffer
            let mut buffer = self
                .device
                .create_host_buffer::<u8>(
                    BufferUsageFlags::TRANSFER_SRC,
                    (img.width() * img.height() * FORMAT_SIZES[&texture.format] as u32) as u64,
                )
                .unwrap();
            buffer.write(0, &img.to_bytes());

            // Create image
            texture.image = Some(
                self.device
                    .create_image_2d(
                        texture.format,
                        self.settings.textures_gen_mipmaps,
                        self.settings.textures_max_anisotropy,
                        ImageUsageFlags::SAMPLED,
                        (img.width(), img.height()),
                    )
                    .unwrap(),
            );
            let image = texture.image.as_ref().unwrap();

            let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

            // Record cmd list: copy resource to image
            {
                let mut cl = self.texture_load_cmd_list.lock().unwrap();
                cl.begin(true).unwrap();
                cl.barrier_image(
                    PipelineStageFlags::TOP_OF_PIPE,
                    PipelineStageFlags::TRANSFER,
                    &[image.barrier_queue(
                        AccessFlags::default(),
                        AccessFlags::TRANSFER_WRITE,
                        ImageLayout::UNDEFINED,
                        ImageLayout::TRANSFER_DST,
                        graphics_queue,
                        graphics_queue,
                    )],
                );
                cl.copy_host_buffer_to_image(&buffer, 0, &image, ImageLayout::TRANSFER_DST);
                cl.barrier_image(
                    PipelineStageFlags::TRANSFER,
                    PipelineStageFlags::BOTTOM_OF_PIPE,
                    &[image.barrier_queue(
                        AccessFlags::TRANSFER_WRITE,
                        AccessFlags::default(),
                        ImageLayout::TRANSFER_DST,
                        ImageLayout::SHADER_READ,
                        graphics_queue,
                        graphics_queue,
                    )],
                );
                cl.end().unwrap();
            }

            // Submit work
            graphics_queue.submit(&mut self.texture_load_packet).unwrap();
            self.texture_load_packet.wait().unwrap();
        }*/
    }

    /// Unload unused texture to free GPU memory
    pub fn unload_texture(&mut self, index: u16) {
        /*if let Some(texture) = self.textures.get_mut(index as usize) {
            texture.loaded = false;
        }*/
    }

    fn acquire_staging_cl(&mut self) -> Result<Arc<Mutex<CmdList>>, vkw::DeviceError> {
        if self.staging_cmd_lists.is_empty() {
            self.device
                .get_queue(Queue::TYPE_GRAPHICS)
                .create_primary_cmd_list()
        } else {
            Ok(self.staging_cmd_lists.remove(self.staging_cmd_lists.len() - 1))
        }
    }

    fn release_staging_cl(&mut self, cl: Arc<Mutex<CmdList>>) {
        self.staging_cmd_lists.push(cl);
    }

    pub fn set_settings(&mut self, settings: Settings) {
        // TODO
        self.settings = settings;
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_size = new_size;
        self.surface_changed = true;

        self.depth_framebuffer = Some(self.depth_render_pass.create_framebuffer(new_size, &[]).unwrap());
    }

    pub fn on_update(&mut self) {
        let mut removed_entity_count = 0;

        // Add new objects to sort
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

        let camera_comp = self.world.read_component::<component::Camera>();
        let camera = camera_comp.get(self.active_camera).unwrap();

        // Sort render objects from front to back (for Z rejection & occlusion queries)
        {
            let transform_comp = self.world.read_component::<component::Transform>();
            let mesh_ref_comp = self.world.read_component::<component::VertexMeshRef>();

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

                    if (a_pos - camera_pos).magnitude() < (b_pos - camera_pos).magnitude() {
                        cmp::Ordering::Less
                    } else {
                        cmp::Ordering::Greater
                    }
                });
            }

            self.sort_count += to_sort_count as u32;
            if self.sort_count >= self.sorted_render_entities.len() as u32 {
                self.sort_count = 0;
            }
        }

        let mut copy_submits = vec![];
        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        let graphics_sp = graphics_queue.get_semaphore();

        // Update camera uniform buffers
        {
            let cam_pos = camera.position();
            let cam_dir = camera.direction();
            let proj = camera.projection();
            let view = camera.view();

            let camera_info = CameraInfo {
                pos: Vector4::new(cam_pos.x, cam_pos.y, cam_pos.z, 0.0),
                dir: Vector4::new(cam_dir.x, cam_dir.y, cam_dir.z, 0.0),
                proj,
                view,
                proj_view: proj * view,
            };

            self.staging_buffer.write(0, unsafe {
                slice::from_raw_parts(
                    &camera_info as *const CameraInfo as *const u8,
                    mem::size_of::<CameraInfo>(),
                )
            });

            // TODO
            /*copy_submits.push(SubmitInfo::new(
                &[WaitSemaphore {
                    semaphore: Arc::clone(&graphics_sp),
                    wait_dst_mask: PipelineStageFlags::BOTTOM_OF_PIPE,
                    wait_value: 0,
                }],
                &[],
            ));*/
        }

        // Update device buffers
        {
            self.staging_copy_submit.set(&copy_submits).unwrap();
            graphics_queue.submit(&mut self.staging_copy_submit).unwrap();
            self.staging_copy_submit.wait().unwrap();
        }
    }

    fn on_render(&mut self, sw_image: &SwapchainImage) -> u64 {
        let camera_component = self.world.read_component::<component::Camera>();
        let active_camera = camera_component.get(self.active_camera).unwrap();

        let transform_comp = self.world.read_component::<component::Transform>();
        let renderer_comp = self.world.read_component::<component::Renderer>();
        let mesh_ref_comp = self.world.read_component::<component::VertexMeshRef>();

        /*let objects: Vec<_> = (&transform_component, &renderer_component, &mesh_ref_component)
        .join()
        .collect();*/
        let object_count = self.sorted_render_entities.len();
        let draw_count_step = object_count / self.secondary_cmd_lists.len() + 1;

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

                cl.bind_pipeline(&self.depth_pipeline_r);
                cl.bind_pipeline_input(0, &self.depth_camera_input);

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

                    if active_camera.is_sphere_visible(center_position, radius) {
                        if renderer.translucent {
                            cl.bind_pipeline(&self.depth_pipeline_r);
                        } else {
                            cl.bind_pipeline(&self.depth_pipeline_rw);
                        }

                        //cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                    }

                    cl.end_query(entity_index as u32);
                }

                cl.end().unwrap();
            });

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);

        // -------------------------------------------------------------------------------------

        let cmd_list = graphics_queue.create_primary_cmd_list().unwrap();

        {
            let mut cmd_list = cmd_list.lock().unwrap();
            cmd_list.begin(true).unwrap();
            cmd_list.begin_render_pass(
                self.rp.as_ref().unwrap(),
                &self.sw_framebuffers[sw_image.get_index() as usize],
                &[ClearValue::ColorF32([0.1, 0.1, 0.1, 1.0])],
                false,
            );

            cmd_list.bind_pipeline(self.pipe.as_ref().unwrap());
            cmd_list.bind_vertex_buffers(0, &[(Arc::clone(&self.vb), 0)]);
            cmd_list.draw(3, 0);

            cmd_list.end_render_pass();

            /*cmd_list.barrier_image(
                PipelineStageFlags::TOP_OF_PIPE,
                PipelineStageFlags::BOTTOM_OF_PIPE,
                &[sw_image.get_image().barrier_queue(
                    AccessFlags::empty(),
                    AccessFlags::empty(),
                    ImageLayout::UNDEFINED,
                    ImageLayout::PRESENT,
                    graphics_queue,
                    graphics_queue,
                )],
            );*/
            cmd_list.end().unwrap();
        }

        let mut packet = self
            .device
            .create_submit_packet(&[SubmitInfo::new(
                &[WaitSemaphore {
                    semaphore: self.swapchain.as_ref().unwrap().get_semaphore(),
                    wait_dst_mask: PipelineStageFlags::TOP_OF_PIPE,
                    wait_value: 0,
                }],
                &[Arc::clone(&cmd_list)],
            )])
            .unwrap();
        graphics_queue.submit(&mut packet).unwrap();

        packet.get_signal_value(0)
    }

    pub fn on_draw(&mut self) {
        let device = self.device.clone();
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

            let swapchain = self.swapchain.clone().unwrap();
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

        self.rp = Some(
            self.device
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
                .unwrap(),
        );

        self.pipe = Some(
            self.device
                .create_graphics_pipeline(
                    self.rp.as_ref().unwrap(),
                    0,
                    PrimitiveTopology::TRIANGLE_LIST,
                    PipelineDepthStencil::new(),
                    PipelineRasterization::new().cull_back_faces(true),
                    &self.sig,
                )
                .unwrap(),
        );

        for img in images {
            self.sw_framebuffers.push(
                self.rp
                    .as_ref()
                    .unwrap()
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
) -> Result<Renderer, vkw::DeviceError> {
    let mut world = specs::World::new();
    world.register::<component::Transform>();
    world.register::<component::VertexMeshRef>();
    world.register::<component::Renderer>();
    world.register::<component::Camera>();
    let renderer_cmp_reader = world.write_component::<component::Renderer>().register_reader();

    // TODO
    let active_camera = world
        .create_entity()
        .with(component::Camera::new(1.0, 90.0, 0.1, 1000.0))
        .build();

    let basic_vertex = device
        .create_shader(
            &resources.get("shaders/basic.vert.spv").unwrap().read().unwrap(),
            &[("inPosition", Format::RGB32_FLOAT)],
            &[],
        )
        .unwrap();
    let basic_pixel = device
        .create_shader(
            &resources.get("shaders/basic.frag.spv").unwrap().read().unwrap(),
            &[],
            &[],
        )
        .unwrap();
    let basic_signature = device
        .create_pipeline_signature(&[basic_vertex, basic_pixel])
        .unwrap();

    let mut vbh = device
        .create_host_buffer::<nalgebra::Vector3<f32>>(BufferUsageFlags::TRANSFER_SRC, 3)
        .unwrap();
    vbh[0] = nalgebra::Vector3::<f32>::new(0.0, -0.5, 0.0);
    vbh[1] = nalgebra::Vector3::<f32>::new(-0.5, 0.5, 0.0);
    vbh[2] = nalgebra::Vector3::<f32>::new(0.5, 0.5, 0.0);

    let vb = device
        .create_device_buffer(
            BufferUsageFlags::VERTEX | BufferUsageFlags::TRANSFER_DST,
            mem::size_of::<nalgebra::Vector3<f32>>() as u64,
            3,
        )
        .unwrap();

    let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);
    let cmd_list = graphics_queue.create_primary_cmd_list().unwrap();
    {
        let mut cmd_list = cmd_list.lock().unwrap();
        cmd_list.begin(true).unwrap();
        cmd_list.copy_buffer_to_device(&vbh, 0, &vb, 0, 3);
        cmd_list.end().unwrap();
    }
    let mut submit = device
        .create_submit_packet(&[SubmitInfo::new(&[], &[cmd_list])])
        .unwrap();
    graphics_queue.submit(&mut submit).unwrap();
    submit.wait().unwrap();

    let texture_load_cmd_list = graphics_queue.create_primary_cmd_list().unwrap();
    let texture_load_packet = device
        .create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&texture_load_cmd_list)])])
        .unwrap();

    let mut secondary_cmd_lists = vec![];
    for _ in 0..num_cpus::get() {
        secondary_cmd_lists.push(graphics_queue.create_secondary_cmd_list()?);
    }

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

    // TODO: dynamic size depending on settings
    let atlas_size = (1024, 1024);

    let texture_atlases = [
        // albedo
        texture_atlas::new(
            &device,
            Format::RGBA8_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            1,
            128,
        )
        .unwrap(),
        // specular
        texture_atlas::new(
            &device,
            Format::RGBA8_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            1,
            128,
        )
        .unwrap(),
        // emission
        texture_atlas::new(
            &device,
            Format::RGBA8_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            1,
            128,
        )
        .unwrap(),
        // normal
        texture_atlas::new(
            &device,
            Format::RGBA16_UNORM,
            settings.textures_gen_mipmaps,
            settings.textures_max_anisotropy,
            1,
            128,
        )
        .unwrap(),
    ];

    let staging_copy_cl = graphics_queue.create_primary_cmd_list()?;
    let staging_copy_submit =
        device.create_submit_packet(&[SubmitInfo::new(&[], &[Arc::clone(&staging_copy_cl)])])?;

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
        //submit_packet: None,
        texture_atlases,
        sig: basic_signature,
        pipe: None,
        rp: None,
        vb,
        staging_buffer: device.create_host_buffer(BufferUsageFlags::TRANSFER_SRC, 0x800000)?,
        staging_cmd_lists: vec![],
        staging_copy_submit,
        sw_framebuffers: vec![],
        query_pool: device.create_query_pool(65536)?,
        secondary_cmd_lists,
        depth_render_pass,
        depth_framebuffer: None,
        depth_pipeline_r,
        depth_pipeline_rw,
        depth_camera_input: depth_signature.create_pool(0, 1)?.get_input(0)?,
        active_camera,
        camera_uniform_buffer: device.create_device_buffer(
            BufferUsageFlags::TRANSFER_DST | BufferUsageFlags::UNIFORM,
            mem::size_of::<CameraInfo>() as u64,
            1,
        )?,
    };
    renderer.on_resize(size);

    Ok(renderer)
}
