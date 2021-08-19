use crate::{Device, Image, ImageView, RenderPass};
use ash::vk;
use std::sync::Arc;

pub struct Framebuffer {
    pub(crate) device: Arc<Device>,
    pub(crate) _render_pass: Arc<RenderPass>,
    pub(crate) native: vk::Framebuffer,
    pub(crate) images: Vec<Option<Arc<Image>>>,
    pub(crate) _image_views: Vec<Arc<ImageView>>,
    pub(crate) size: (u32, u32),
}

impl Framebuffer {
    /// May return [`None`] if attachment at `index` was overridden by image view at framebuffer creation.
    pub fn get_image(&self, index: u32) -> Option<&Arc<Image>> {
        self.images.get(index as usize)?.as_ref()
    }
}

impl Drop for Framebuffer {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.native.destroy_framebuffer(self.native, None) };
    }
}
