use crate::{Device, Image, RenderPass};
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::Arc;

pub struct Framebuffer {
    pub(crate) device: Arc<Device>,
    pub(crate) render_pass: Arc<RenderPass>,
    pub(crate) native: vk::Framebuffer,
    pub(crate) images: Vec<Arc<Image>>,
    pub(crate) size: (u32, u32),
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_framebuffer(self.native, None) };
    }
}
