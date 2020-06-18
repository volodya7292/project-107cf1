pub(crate) mod component;
mod material_pipeline;
mod scene;
mod vertex_mesh;

use crate::renderer::scene::Scene;
use std::rc::Rc;
use std::sync::Arc;
use vk_wrapper as vkw;

pub struct Renderer {
    surface: Rc<vkw::Surface>,
    swapchain: Rc<vkw::Swapchain>,
    surface_changed: bool,
    surface_size: (u32, u32),
    vsync: bool,
    device: Arc<vkw::Device>,
    scene: Scene,
    submit_packet: Option<vkw::SubmitPacket>,
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

        cmd_list.begin(true).unwrap();
        cmd_list.barrier_image(
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
        );
        cmd_list.end().unwrap();

        self.submit_packet = Some(
            self.device
                .create_submit_packet(&[vkw::SubmitInfo::new(
                    &[vkw::WaitSemaphore {
                        semaphore: self.swapchain.get_semaphore(),
                        wait_dst_mask: vkw::PipelineStageFlags::TOP_OF_PIPE,
                        wait_value: 0,
                    }],
                    &[Rc::clone(&cmd_list)],
                )])
                .unwrap(),
        );
        graphics_queue
            .submit(&mut self.submit_packet.as_mut().unwrap())
            .unwrap();

        self.submit_packet.as_mut().unwrap().get_signal_value(0)
    }

    pub fn on_draw(&mut self) {
        let device = self.device.clone();
        let adapter = device.get_adapter();
        let graphics_queue = device.get_queue(vkw::Queue::TYPE_GRAPHICS);
        let surface = &self.surface;

        if adapter.is_surface_valid(surface).unwrap() {
            if self.surface_changed {
                self.swapchain = device
                    .create_swapchain(&self.surface, self.surface_size, true)
                    .unwrap();
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
}

pub fn new(
    surface: &Rc<vkw::Surface>,
    size: (u32, u32),
    vsync: bool,
    device: &Arc<vkw::Device>,
) -> Result<Renderer, vkw::DeviceError> {
    Ok(Renderer {
        surface: Rc::clone(surface),
        swapchain: device.create_swapchain(surface, size, vsync)?,
        surface_changed: false,
        surface_size: size,
        vsync,
        device: Arc::clone(device),
        scene: scene::new(),
        submit_packet: None,
    })
}
