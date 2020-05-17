use crate::{Semaphore, device::Device, Image};
use ash::vk;
use std::rc::Rc;

pub struct Swapchain {
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::SwapchainKHR,
    pub(crate) images: Vec<Image>,
}

impl Swapchain {
    pub fn acquire_next_image(&self, semaphore: &Rc<Semaphore>) -> Result<(u32, bool), vk::Result> {
        Ok(unsafe {
            self.device.swapchain_khr.acquire_next_image(
                self.native,
                u64::MAX,
                semaphore.native,
                vk::Fence::default(),
            )?
        })
    }

    pub fn get_image(&self, index: u32) -> Option<&Image> {
        self.images.get(index as usize)
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.device.swapchain_khr.destroy_swapchain(self.native, None) };
    }
}
