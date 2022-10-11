use std::sync::Arc;

use ash::vk;
use parking_lot::Mutex;

use crate::{Device, Image, Semaphore, Surface};

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
    pub(crate) _surface: Arc<Surface>,
    pub(crate) native: Mutex<vk::SwapchainKHR>,
}

impl Drop for SwapchainWrapper {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            self.device
                .swapchain_khr
                .destroy_swapchain(*self.native.lock(), None);
        }
    }
}

pub struct Swapchain {
    pub(crate) wrapper: Arc<SwapchainWrapper>,
    pub(crate) readiness_semaphore: Arc<Semaphore>,
    pub(crate) images: Vec<Arc<Image>>,
}

impl Swapchain {
    pub fn images(&self) -> &[Arc<Image>] {
        &self.images
    }

    pub fn acquire_image(&self) -> Result<(SwapchainImage, bool), Error> {
        let result = unsafe {
            let native = self.wrapper.native.lock();
            self.wrapper.device.swapchain_khr.acquire_next_image(
                *native,
                u64::MAX,
                self.readiness_semaphore.native,
                vk::Fence::default(),
            )
        };

        match result {
            Ok(a) => Ok((
                SwapchainImage {
                    image: Arc::clone(&self.images[a.0 as usize]),
                    index: a.0,
                },
                a.1,
            )),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::IncompatibleSurface),
            Err(e) => Err(Error::VkError(e)),
        }
    }

    pub fn readiness_semaphore(&self) -> &Arc<Semaphore> {
        &self.readiness_semaphore
    }
}

pub struct SwapchainImage {
    pub(crate) image: Arc<Image>,
    pub(crate) index: u32,
}

impl SwapchainImage {
    pub fn get(&self) -> &Arc<Image> {
        &self.image
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}
