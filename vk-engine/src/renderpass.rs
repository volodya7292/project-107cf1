use ash::vk;

pub struct RenderPass {
    pub(crate) native: vk::RenderPass,
}