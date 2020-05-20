use ash::vk;

pub struct CmdList {
    pub(crate) native: vk::CommandBuffer,
}
