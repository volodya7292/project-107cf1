use ash::vk;

pub struct Framebuffer {
    pub(crate) native: vk::Framebuffer,
}