pub(crate) mod component;
pub(crate) mod material_pipeline;
mod texture;
#[macro_use]
pub(crate) mod vertex_mesh;

use crate::renderer::vertex_mesh::VertexMeshCmdList;
use crate::resource_file::{ResourceFile, ResourceRef};
use image::GenericImageView;
use rayon::prelude::*;
use specs::prelude::ParallelIterator;
use specs::WorldExt;
use specs::{Builder, Join};
use std::mem;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;
use vk_wrapper::{
    Attachment, AttachmentRef, BufferUsageFlags, ClearValue, CmdList, DeviceBuffer, Format, Framebuffer,
    ImageLayout, ImageMod, LoadStore, Pipeline, PipelineDepthStencil, PipelineRasterization,
    PipelineSignature, PrimitiveTopology, QueryPool, Queue, QueueType, RenderPass, SubmitInfo, SubmitPacket,
    Subpass, FORMAT_SIZES,
};

pub struct Renderer {
    world: specs::World,

    surface: Arc<vkw::Surface>,
    swapchain: Option<Arc<vkw::Swapchain>>,
    surface_changed: bool,
    surface_size: (u32, u32),
    settings: Settings,
    device: Arc<vkw::Device>,

    textures: Vec<Texture>,
    texture_load_cmd_list: Arc<Mutex<vkw::CmdList>>,
    texture_load_packet: vkw::SubmitPacket,

    //submit_packet: Option<vkw::SubmitPacket>,
    sig: Arc<vkw::PipelineSignature>,
    pipe: Option<Arc<vkw::Pipeline>>,
    rp: Option<Arc<vkw::RenderPass>>,
    vb: Arc<vkw::DeviceBuffer>,

    main_framebuffers: Vec<Arc<vkw::Framebuffer>>,

    query_pool: Arc<QueryPool>,
    secondary_cmd_lists: Vec<Arc<Mutex<CmdList>>>,

    depth_render_pass: Arc<RenderPass>,
    depth_framebuffer: Option<Arc<Framebuffer>>,
    depth_pipeline_r: Arc<Pipeline>,
    depth_pipeline_rw: Arc<Pipeline>,

    active_camera: specs::Entity,
}

#[derive(Copy, Clone)]
pub struct Settings {
    pub(crate) vsync: bool,
    pub(crate) textures_gen_mipmaps: bool,
    pub(crate) textures_max_anisotropy: f32,
}

pub struct Texture {
    res_ref: ResourceRef,
    image: Option<Arc<vkw::Image>>,
    format: vkw::Format,
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
    pub fn add_texture(&mut self, res_ref: ResourceRef, format: vkw::Format) -> u16 {
        self.textures.push(Texture {
            res_ref,
            image: None,
            format,
        });
        (self.textures.len() - 1) as u16
    }

    /// Texture must be loaded before use in a shader
    pub fn load_texture(&mut self, index: u16) {
        if let Some(texture) = self.textures.get_mut(index as usize) {
            // Load resource
            let bytes = texture.res_ref.read().unwrap();
            let img = image::load_from_memory(&bytes).unwrap();

            // Create staging buffer
            let mut buffer = self
                .device
                .create_host_buffer::<u8>(
                    vkw::BufferUsageFlags::TRANSFER_SRC,
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
                        vkw::ImageUsageFlags::SAMPLED,
                        (img.width(), img.height()),
                    )
                    .unwrap(),
            );
            let image = texture.image.as_ref().unwrap();

            let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);

            // Record cmd list: copy resource to image
            {
                let mut cl = self.texture_load_cmd_list.lock().unwrap();
                cl.begin(true).unwrap();
                cl.barrier_image(
                    vkw::PipelineStageFlags::TOP_OF_PIPE,
                    vkw::PipelineStageFlags::TRANSFER,
                    &[image.barrier_queue(
                        vkw::AccessFlags::default(),
                        vkw::AccessFlags::TRANSFER_WRITE,
                        vkw::ImageLayout::UNDEFINED,
                        vkw::ImageLayout::TRANSFER_DST,
                        graphics_queue,
                        graphics_queue,
                    )],
                );
                cl.copy_host_buffer_to_image(&buffer, 0, &image, vkw::ImageLayout::TRANSFER_DST);
                cl.barrier_image(
                    vkw::PipelineStageFlags::TRANSFER,
                    vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                    &[image.barrier_queue(
                        vkw::AccessFlags::TRANSFER_WRITE,
                        vkw::AccessFlags::default(),
                        vkw::ImageLayout::TRANSFER_DST,
                        vkw::ImageLayout::SHADER_READ,
                        graphics_queue,
                        graphics_queue,
                    )],
                );
                cl.end().unwrap();
            }

            // Submit work
            graphics_queue.submit(&mut self.texture_load_packet).unwrap();
            self.texture_load_packet.wait().unwrap();
        }
    }

    /// Unload unused texture to free GPU memory
    pub fn unload_texture(&mut self, index: u16) {
        if let Some(texture) = self.textures.get_mut(index as usize) {
            texture.image = None;
        }
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
        self.world.maintain();
    }

    fn on_render(&mut self, sw_image: &vkw::SwapchainImage) -> u64 {
        let camera_component = self.world.read_component::<component::Camera>();
        let active_camera = camera_component.get(self.active_camera).unwrap();

        let transform_component = self.world.read_component::<component::Transform>();
        let renderer_component = self.world.read_component::<component::Renderer>();
        let mesh_ref_component = self.world.read_component::<component::VertexMeshRef>();

        let objects: Vec<_> = (&transform_component, &renderer_component, &mesh_ref_component)
            .join()
            .collect();
        let object_count = objects.len();
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

                for j in 0..draw_count_step {
                    let entity_index = i * draw_count_step + j;
                    if entity_index >= object_count {
                        break;
                    }

                    let (transform, renderer, mesh_ref) = objects[entity_index];

                    let vertex_mesh = mesh_ref.vertex_mesh();
                    let aabb = {
                        let vertex_mesh = vertex_mesh.lock().unwrap();
                        vertex_mesh.aabb().clone()
                    };

                    let center_position = (aabb.0 + aabb.1) * 0.5 + transform.position();
                    let radius = ((aabb.1 - aabb.0).component_mul(transform.scale()) * 0.5).magnitude();

                    cl.begin_query(&self.query_pool, entity_index as u32);

                    if active_camera.is_sphere_visible(center_position, radius) {
                        //cl.bind_pipeline(pipeline);
                        //cl.bind_and_draw_vertex_mesh(&vertex_mesh);
                    }

                    cl.end_query(entity_index as u32);
                }

                cl.end().unwrap();
            });

        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);

        // -------------------------------------------------------------------------------------

        let cmd_list = graphics_queue.create_primary_cmd_list().unwrap();

        {
            let mut cmd_list = cmd_list.lock().unwrap();
            cmd_list.begin(true).unwrap();
            cmd_list.begin_render_pass(
                self.rp.as_ref().unwrap(),
                &self.main_framebuffers[sw_image.get_index() as usize],
                &[ClearValue::ColorF32([0.1, 0.1, 0.1, 1.0])],
                false,
            );

            cmd_list.bind_pipeline(self.pipe.as_ref().unwrap());
            cmd_list.bind_vertex_buffers(0, &[(Arc::clone(&self.vb), 0)]);
            cmd_list.draw(3, 0);

            cmd_list.end_render_pass();

            /*cmd_list.barrier_image(
                vkw::PipelineStageFlags::TOP_OF_PIPE,
                vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                &[sw_image.get_image().barrier_queue(
                    vkw::AccessFlags::empty(),
                    vkw::AccessFlags::empty(),
                    vkw::ImageLayout::UNDEFINED,
                    vkw::ImageLayout::PRESENT,
                    graphics_queue,
                    graphics_queue,
                )],
            );*/
            cmd_list.end().unwrap();
        }

        let mut packet = self
            .device
            .create_submit_packet(&[vkw::SubmitInfo::new(
                &[vkw::WaitSemaphore {
                    semaphore: self.swapchain.as_ref().unwrap().get_semaphore(),
                    wait_dst_mask: vkw::PipelineStageFlags::TOP_OF_PIPE,
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
        let graphics_queue = device.get_queue(vkw::Queue::TYPE_GRAPHICS);
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

                let present_queue = self.device.get_queue(vkw::Queue::TYPE_PRESENT);
                if !present_queue
                    .present(sw_image, graphics_queue.get_semaphore(), wait_value)
                    .unwrap_or(false)
                {
                    self.surface_changed = true;
                }
            } else {
                match acquire_result {
                    Err(vkw::swapchain::Error::IncompatibleSurface) => {
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
        self.main_framebuffers.clear();

        let images = self.swapchain.as_ref().unwrap().get_images();

        self.rp = Some(
            self.device
                .create_render_pass(
                    &[Attachment {
                        format: images[0].get_format(),
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
            self.main_framebuffers.push(
                self.rp
                    .as_ref()
                    .unwrap()
                    .create_framebuffer(
                        self.surface_size,
                        &[(0, ImageMod::OverrideImage(Arc::clone(img)))],
                    )
                    .unwrap(),
            );
        }
    }
}

pub fn new(
    surface: &Arc<vkw::Surface>,
    size: (u32, u32),
    settings: Settings,
    device: &Arc<vkw::Device>,
    resources: &Arc<ResourceFile>,
) -> Result<Renderer, vkw::DeviceError> {
    let mut world = specs::World::new();
    world.register::<component::Transform>();
    world.register::<component::VertexMeshRef>();
    world.register::<component::Renderer>();
    world.register::<component::Camera>();

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
        &[vkw::Attachment {
            format: vkw::Format::D32_FLOAT,
            init_layout: vkw::ImageLayout::UNDEFINED,
            final_layout: vkw::ImageLayout::DEPTH_READ,
            load_store: LoadStore::InitClearFinalSave,
        }],
        &[vkw::Subpass {
            color: vec![],
            depth: Some(AttachmentRef {
                index: 0,
                layout: vkw::ImageLayout::DEPTH_STENCIL_ATTACHMENT,
            }),
        }],
        &[],
    )?;
    let depth_vertex = device.create_shader(
        &resources.get("shaders/depth.vert.spv").unwrap().read().unwrap(),
        &[("inPosition", vkw::Format::RGB32_FLOAT)],
        &[],
    )?;
    let depth_signature = device.create_pipeline_signature(&[depth_vertex])?;
    let depth_pipeline_r = device.create_graphics_pipeline(
        &depth_render_pass,
        0,
        vkw::PrimitiveTopology::TRIANGLE_LIST,
        vkw::PipelineDepthStencil::new()
            .depth_test(true)
            .depth_write(false),
        vkw::PipelineRasterization::new().cull_back_faces(true),
        &depth_signature,
    )?;
    let depth_pipeline_rw = device.create_graphics_pipeline(
        &depth_render_pass,
        0,
        vkw::PrimitiveTopology::TRIANGLE_LIST,
        vkw::PipelineDepthStencil::new()
            .depth_test(true)
            .depth_write(true),
        vkw::PipelineRasterization::new().cull_back_faces(true),
        &depth_signature,
    )?;

    let mut renderer = Renderer {
        world,
        surface: Arc::clone(surface),
        swapchain: None,
        surface_changed: false,
        surface_size: size,
        settings,
        device: Arc::clone(device),
        //submit_packet: None,
        textures: Vec::with_capacity(65536),
        texture_load_cmd_list,
        texture_load_packet,
        sig: basic_signature,
        pipe: None,
        rp: None,
        vb,
        main_framebuffers: vec![],
        query_pool: device.create_query_pool(65536)?,
        secondary_cmd_lists,
        depth_render_pass,
        depth_framebuffer: None,
        depth_pipeline_r,
        depth_pipeline_rw,
        active_camera,
    };
    renderer.on_resize(size);

    Ok(renderer)
}
