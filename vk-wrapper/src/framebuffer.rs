use crate::{Device, Image, RenderPass};
use ash::version::DeviceV1_0;
use ash::vk;
use std::rc::Rc;
use std::sync::Arc;

pub struct Framebuffer {
    pub(crate) device: Arc<Device>,
    pub(crate) renderpass: Arc<RenderPass>,
    pub(crate) native: vk::Framebuffer,
    pub(crate) images: Vec<Rc<Image>>,
    pub(crate) size: (u32, u32),
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_framebuffer(self.native, None) };
    }
}
