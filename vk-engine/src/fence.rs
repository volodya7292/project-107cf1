use crate::Device;
use ash::version::DeviceV1_0;
use ash::vk;
use std::{rc::Rc, slice};

pub struct Fence {
    pub(crate) native_device: Rc<ash::Device>,
    pub(crate) native: vk::Fence,
}

impl Fence {
    pub(crate) fn reset(&self) -> Result<(), vk::Result> {
        unsafe { self.native_device.reset_fences(slice::from_ref(&self.native)) }
    }

    pub(crate) fn wait(&self) -> Result<(), vk::Result> {
        unsafe {
            self.native_device
                .wait_for_fences(slice::from_ref(&self.native), true, u64::MAX)
        }
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        self.wait().unwrap();
        unsafe { self.native_device.destroy_fence(self.native, None) };
    }
}
