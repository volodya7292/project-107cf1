use crate::{Device, Image, Semaphore};
use ash::vk;
use std::cell::Cell;
use std::rc::Rc;

pub struct Swapchain {
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::SwapchainKHR,
    pub(crate) semaphore: Semaphore,
    pub(crate) images: Vec<Image>,
    pub(crate) curr_image: Cell<Option<(u32, bool)>>,
}

impl Swapchain {
    pub fn acquire_image(self: &Rc<Self>) -> Result<(SwapchainImage, bool), vk::Result> {
        if self.curr_image.get().is_none() {
            self.curr_image.set(Some(unsafe {
                self.device.swapchain_khr.acquire_next_image(
                    self.native,
                    u64::MAX,
                    self.semaphore.native,
                    vk::Fence::default(),
                )?
            }));
        }

        let (index, success) = self.curr_image.get().unwrap();

        Ok((
            SwapchainImage {
                swapchain: Rc::clone(self),
                index: index,
            },
            success,
        ))
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

pub struct SwapchainImage {
    pub(crate) swapchain: Rc<Swapchain>,
    pub(crate) index: u32,
}
