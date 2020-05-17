use ash::vk;

pub struct Fence {
    native: vk::Fence,
}