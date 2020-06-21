pub(crate) mod component;
mod material_pipeline;
mod scene;
mod vertex_mesh;

use crate::renderer::scene::Scene;
use crate::resource_file::ResourceFile;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;
use vk_wrapper::{
    Attachment, AttachmentRef, BufferUsageFlags, ClearValue, DeviceBuffer, Format, Framebuffer, ImageLayout,
    ImageMod, LoadStore, Pipeline, PipelineDepthStencil, PipelineRasterization, PipelineSignature,
    PrimitiveTopology, Queue, QueueType, RenderPass, SubmitInfo, Subpass,
};

pub struct Renderer {
    surface: Rc<vkw::Surface>,
    swapchain: Rc<vkw::Swapchain>,
    surface_changed: bool,
    surface_size: (u32, u32),
    vsync: bool,
    device: Arc<vkw::Device>,
    scene: Scene,
    //submit_packet: Option<vkw::SubmitPacket>,
    sig: Arc<PipelineSignature>,
    pipe: Option<Arc<Pipeline>>,
    rp: Option<Arc<RenderPass>>,
    main_framebuffers: Vec<Arc<Framebuffer>>,
    vb: Arc<DeviceBuffer>,
}

impl Renderer {
    /*fn set_scene(&mut self, scene: Scene) {
        self.scene = Some(scene);
    }*/

    pub fn scene(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn on_resize(&mut self, new_size: (u32, u32)) {
        self.surface_size = new_size;
        self.surface_changed = true;
    }

    pub fn on_update(&self) {}

    fn on_render(&mut self, sw_image: &vkw::SwapchainImage) -> u64 {
        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);
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
            cmd_list.bind_vertex_buffer(0, &[(Arc::clone(&self.vb), 0)]);
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
                    semaphore: self.swapchain.get_semaphore(),
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
                self.swapchain = device
                    .create_swapchain(&self.surface, self.surface_size, self.vsync)
                    .unwrap();
                self.create_main_framebuffers();
                self.surface_changed = false;
            }

            let swapchain = self.swapchain.clone();
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

        let images = self.swapchain.get_images();

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
    surface: &Rc<vkw::Surface>,
    size: (u32, u32),
    vsync: bool,
    device: &Arc<vkw::Device>,
    resources: &mut ResourceFile,
) -> Result<Renderer, vkw::DeviceError> {
    let basic_vertex = device
        .create_shader(
            &resources.read("shaders/basic.vert.spv").unwrap(),
            &[("inPosition", Format::RGB32_FLOAT)],
        )
        .unwrap();
    let basic_pixel = device
        .create_shader(&resources.read("shaders/basic.frag.spv").unwrap(), &[])
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
        .create_device_buffer::<nalgebra::Vector3<f32>>(
            BufferUsageFlags::VERTEX | BufferUsageFlags::TRANSFER_DST,
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

    let mut renderer = Renderer {
        surface: Rc::clone(surface),
        swapchain: device.create_swapchain(surface, size, vsync)?,
        surface_changed: false,
        surface_size: size,
        vsync,
        device: Arc::clone(device),
        scene: scene::new(),
        //submit_packet: None,
        sig: basic_signature,
        pipe: None,
        rp: None,
        main_framebuffers: vec![],
        vb,
    };
    renderer.create_main_framebuffers();

    Ok(renderer)
}
