use crate::{Device, Image, Semaphore, Surface};
use ash::vk;
use std::cell::Cell;
use std::rc::Rc;
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
    pub(crate) native: vk::SwapchainKHR,
}

impl Drop for SwapchainWrapper {
    fn drop(&mut self) {
        unsafe { self.device.swapchain_khr.destroy_swapchain(self.native, None) };
    }
}

pub struct Swapchain {
    pub(crate) wrapper: Rc<SwapchainWrapper>,
    pub(crate) _surface: Rc<Surface>,
    pub(crate) semaphore: Rc<Semaphore>,
    pub(crate) images: Vec<Rc<Image>>,
    pub(crate) curr_image: Cell<Option<(u32, bool)>>,
}

impl Swapchain {
    pub fn acquire_image(&self) -> Result<(SwapchainImage, bool), Error> {
        if self.curr_image.get().is_none() {
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
                    self.curr_image.set(Some((a.0, !a.1)));
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    return Err(Error::IncompatibleSurface);
                }
                Err(e) => {
                    return Err(Error::VkError(e));
                }
            };
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

pub struct SwapchainImage<'a> {
    pub(crate) swapchain: &'a Swapchain,
    pub(crate) index: u32,
}

impl SwapchainImage<'_> {
    pub fn get_image(&self) -> &Rc<Image> {
        &self.swapchain.images[self.index as usize]
    }
}
