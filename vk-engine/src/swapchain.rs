use crate::{Device, Image, Semaphore};
use ash::vk;
use std::cell::Cell;
use std::rc::Rc;

pub struct Swapchain {
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::SwapchainKHR,
    pub(crate) semaphore: Rc<Semaphore>,
    pub(crate) images: Vec<Image>,
    pub(crate) curr_image: Cell<Option<(u32, bool)>>,
}

impl Swapchain {
    pub fn acquire_image(&self) -> Result<(SwapchainImage, bool), vk::Result> {
        if self.curr_image.get().is_none() {
            let (index, success) = unsafe {
                self.device.swapchain_khr.acquire_next_image(
                    self.native,
                    u64::MAX,
                    self.semaphore.native,
                    vk::Fence::default(),
                )?
            };
            self.curr_image.set(Some((index, !success)));
        }

        let (index, optimal) = self.curr_image.get().unwrap();
        Ok((
            SwapchainImage {
                swapchain: self,
                index,
            },
            optimal,
        ))
    }

    pub fn get_semaphore(&self) -> Rc<Semaphore> {
        Rc::clone(&self.semaphore)
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe { self.device.swapchain_khr.destroy_swapchain(self.native, None) };
    }
}

pub struct SwapchainImage<'a> {
    pub(crate) swapchain: &'a Swapchain,
    pub(crate) index: u32,
}

impl SwapchainImage<'_> {
    pub fn get_image(&self) -> &Image {
        &self.swapchain.images[self.index as usize]
    }
}
