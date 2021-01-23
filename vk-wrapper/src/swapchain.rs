use crate::{Device, Image, Semaphore, Surface};
use ash::version::DeviceV1_0;
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
    pub(crate) native: Mutex<vk::SwapchainKHR>,
}

impl Drop for SwapchainWrapper {
    fn drop(&mut self) {
        unsafe {
            self.device.wrapper.0.device_wait_idle().unwrap();
            self.device
                .swapchain_khr
                .destroy_swapchain(*self.native.lock().unwrap(), None);
        }
    }
}

pub struct Swapchain {
    pub(crate) wrapper: Arc<SwapchainWrapper>,
    pub(crate) _surface: Arc<Surface>,
    pub(crate) semaphore: Arc<Semaphore>,
    pub(crate) images: Vec<Arc<Image>>,
}

impl Swapchain {
    pub fn get_images(&self) -> &Vec<Arc<Image>> {
        &self.images
    }

    pub fn acquire_image(&self) -> Result<(SwapchainImage, bool), Error> {
        let result = unsafe {
            let native = self.wrapper.native.lock().unwrap();
            self.wrapper.device.swapchain_khr.acquire_next_image(
                *native,
                u64::MAX,
                self.semaphore.native,
                vk::Fence::default(),
            )
        };

        match result {
            Ok(a) => Ok((
                SwapchainImage {
                    swapchain: self,
                    index: a.0,
                },
                a.1,
            )),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::IncompatibleSurface),
            Err(e) => Err(Error::VkError(e)),
        }
    }

    pub fn get_semaphore(&self) -> &Arc<Semaphore> {
        &self.semaphore
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
