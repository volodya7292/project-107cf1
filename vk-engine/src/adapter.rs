use ash::vk;

pub struct Adapter {
    pub(crate) native: vk::PhysicalDevice,
}
