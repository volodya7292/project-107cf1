use ash::vk;

pub struct Framebuffer {
    pub(crate) native: vk::Framebuffer,
    pub(crate) size: (u32, u32),
}