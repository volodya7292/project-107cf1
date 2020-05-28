use crate::device::DeviceWrapper;
use ash::version::DeviceV1_0;
use ash::vk;
use std::cell::Cell;
use std::{rc::Rc, slice};

pub struct Fence {
    pub(crate) device_wrapper: Rc<DeviceWrapper>,
    pub(crate) native: vk::Fence,
}

impl Fence {
    pub(crate) fn reset(&self) -> Result<(), vk::Result> {
        self.wait()?;
        unsafe { self.device_wrapper.0.reset_fences(slice::from_ref(&self.native)) }
    }

    pub(crate) fn wait(&self) -> Result<(), vk::Result> {
        unsafe {
            self.device_wrapper
                .0
                .wait_for_fences(slice::from_ref(&self.native), true, u64::MAX)
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        self.wait().unwrap();
        unsafe { self.device_wrapper.0.destroy_fence(self.native, None) };
    }
}
