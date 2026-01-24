use crate::{Device, Fence, Image, Semaphore, Surface};
use ash::vk;
use common::parking_lot::Mutex;
use std::sync::Arc;

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
    pub(crate) fence: Mutex<Fence>,
    pub(crate) images: Vec<Arc<Image>>,
    pub(crate) image_completion_semaphores: Vec<Arc<Semaphore>>,
}

impl Swapchain {
    pub fn images(&self) -> &[Arc<Image>] {
        &self.images
    }

    pub fn acquire_image(&self) -> Result<(SwapchainImage, bool), Error> {
        let mut fence = self.fence.lock();
        fence.reset().unwrap();

        let result = unsafe {
            let native = self.wrapper.native.lock();
            self.wrapper.device.swapchain_khr.acquire_next_image(
                *native,
                u64::MAX,
                vk::Semaphore::null(),
                fence.native,
            )
        };
        fence.wait().unwrap();

        match result {
            Ok(a) => Ok((
                SwapchainImage {
                    image: Arc::clone(&self.images[a.0 as usize]),
                    completion_semaphore: Arc::clone(&self.image_completion_semaphores[a.0 as usize]),
                    index: a.0,
                },
                a.1,
            )),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(Error::IncompatibleSurface),
            Err(e) => Err(Error::VkError(e)),
        }
    }
}

impl PartialEq for Swapchain {
    fn eq(&self, other: &Self) -> bool {
        Arc::as_ptr(&self.wrapper) == Arc::as_ptr(&other.wrapper)
    }
}

pub struct SwapchainImage {
    pub(crate) image: Arc<Image>,
    pub(crate) completion_semaphore: Arc<Semaphore>,
    pub(crate) index: u32,
}

impl SwapchainImage {
    pub fn get(&self) -> &Arc<Image> {
        &self.image
    }

    pub fn completion_semaphore(&self) -> &Arc<Semaphore> {
        &self.completion_semaphore
    }

    pub fn index(&self) -> u32 {
        self.index
    }
}
