use ash::vk;

pub struct CmdList {
    native: vk::CommandBuffer,
}