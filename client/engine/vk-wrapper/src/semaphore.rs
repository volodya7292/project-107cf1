use crate::device::DeviceWrapper;
use crate::{Device, DeviceError};
use ash::vk;
use std::slice;
use std::sync::Arc;

pub struct Semaphore {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) native: vk::Semaphore,
    pub(crate) semaphore_type: vk::SemaphoreTypeKHR,
}

impl Semaphore {
    pub fn get_current_value(&self) -> Result<u64, vk::Result> {
        unsafe {
            self.device_wrapper
                .native
                .get_semaphore_counter_value(self.native)
        }
    }

    pub fn wait(&self, value: u64) -> Result<(), DeviceError> {
        if self.semaphore_type != vk::SemaphoreTypeKHR::TIMELINE {
            panic!("Semaphore type must be TIMELINE!");
        }

        let wait_info = vk::SemaphoreWaitInfoKHR::builder()
            .semaphores(slice::from_ref(&self.native))
            .values(slice::from_ref(&value));

        Ok(unsafe { self.device_wrapper.native.wait_semaphores(&wait_info, u64::MAX)? })
    }

    pub fn signal(&self, value: u64) -> Result<(), DeviceError> {
        if self.semaphore_type != vk::SemaphoreTypeKHR::TIMELINE {
            panic!("Semaphore type must be TIMELINE!");
        }

        let signal_info = vk::SemaphoreSignalInfoKHR::builder()
            .semaphore(self.native)
            .value(value);

        Ok(unsafe { self.device_wrapper.native.signal_semaphore(&signal_info)? })
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe { self.device_wrapper.native.destroy_semaphore(self.native, None) };
    }
}

impl Device {
    pub fn wait_semaphores(&self, semaphores: &[&Semaphore], values: &[u64]) -> Result<(), vk::Result> {
        let natives: Vec<_> = semaphores.iter().map(|v| v.native).collect();

        let wait_info = vk::SemaphoreWaitInfoKHR::builder()
            .semaphores(&natives)
            .values(values);

        unsafe { self.wrapper.native.wait_semaphores(&wait_info, u64::MAX) }
    }
}
