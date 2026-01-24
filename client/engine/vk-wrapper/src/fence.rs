use std::sync::Arc;

use ash::vk;

use crate::device::DeviceWrapper;

pub struct Fence {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) native: vk::Fence,
}

impl Fence {
    pub(crate) fn reset(&mut self) -> Result<(), vk::Result> {
        unsafe { self.device_wrapper.native.reset_fences(&[self.native]) }
    }

    pub(crate) fn wait(&self) -> Result<(), vk::Result> {
        unsafe {
            self.device_wrapper
                .native
                .wait_for_fences(&[self.native], true, u64::MAX)
        }
    }

    pub(crate) fn wait_and_reset(&mut self) -> Result<(), vk::Result> {
        self.wait()?;
        self.reset()
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        self.wait().unwrap();
        unsafe { self.device_wrapper.native.destroy_fence(self.native, None) };
    }
}
