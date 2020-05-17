use ash::vk;

pub struct Semaphore {
    pub(crate) native: vk::Semaphore,
}