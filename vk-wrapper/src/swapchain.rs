use crate::{Device, Image, Semaphore, Surface};
use ash::vk;
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub enum Error {
    VkError(vk::Result),
    IncompatibleSurface,
}

impl From<vk::Result> for Error {
    fn from(err: vk::Result) -> Self {
        Self::VkError(err)
    }
}

pub(crate) struct SwapchainWrapper {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::SwapchainKHR,
}

impl Drop for SwapchainWrapper {
    fn drop(&mut self) {
        unsafe { self.device.swapchain_khr.destroy_swapchain(self.native, None) };
    }
}

pub struct Swapchain {
    pub(crate) wrapper: Arc<SwapchainWrapper>,
    pub(crate) _surface: Arc<Surface>,
    pub(crate) semaphore: Arc<Semaphore>,
    pub(crate) images: Vec<Arc<Image>>,
    pub(crate) curr_image: Mutex<Option<(u32, bool)>>,
}

impl Swapchain {
    pub fn get_images(&self) -> &Vec<Arc<Image>> {
        &self.images
    }

    pub fn acquire_image(&self) -> Result<(SwapchainImage, bool), Error> {
        let mut curr_image = self.curr_image.lock().unwrap();

        if curr_image.is_none() {
            let result = unsafe {
                self.wrapper.device.swapchain_khr.acquire_next_image(
                    self.wrapper.native,
                    u64::MAX,
                    self.semaphore.native,
                    vk::Fence::default(),
                )
            };
            match result {
                Ok(a) => {
                    *curr_image = Some((a.0, !a.1));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Err(Error::IncompatibleSurface);
                }
                Err(e) => {
                    return Err(Error::VkError(e));
                }
            };
        }

        let (index, optimal) = curr_image.unwrap();
        Ok((
            SwapchainImage {
                swapchain: self,
                index,
            },
            optimal,
        ))
    }

    pub fn get_semaphore(&self) -> Arc<Semaphore> {
        Arc::clone(&self.semaphore)
    }
}

pub struct SwapchainImage<'a> {
    pub(crate) swapchain: &'a Swapchain,
    pub(crate) index: u32,
}

impl SwapchainImage<'_> {
    pub fn get_image(&self) -> &Arc<Image> {
        &self.swapchain.images[self.index as usize]
    }

    pub fn get_index(&self) -> u32 {
        self.index
    }
}
