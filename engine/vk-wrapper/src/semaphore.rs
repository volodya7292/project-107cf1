use std::slice;
use std::sync::Arc;

use ash::vk;
use parking_lot::Mutex;

use crate::device::DeviceWrapper;

pub struct Semaphore {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) native: vk::Semaphore,
    pub(crate) semaphore_type: vk::SemaphoreTypeKHR,
    pub(crate) last_signal_value: Mutex<u64>,
}

impl Semaphore {
    pub fn get_current_value(&self) -> Result<u64, vk::Result> {
        unsafe {
            self.device_wrapper
                .ts_khr
                .get_semaphore_counter_value(self.native)
        }
    }

    pub fn wait(&self, value: u64) -> Result<(), vk::Result> {
        if self.semaphore_type != vk::SemaphoreTypeKHR::TIMELINE {
            panic!("Semaphore type is not TIMELINE!");
        }

        let wait_info = vk::SemaphoreWaitInfoKHR::builder()
            .semaphores(slice::from_ref(&self.native))
            .values(slice::from_ref(&value));

        unsafe { self.device_wrapper.ts_khr.wait_semaphores(&wait_info, u64::MAX) }
    }

    pub fn signal(&self) -> Result<u64, vk::Result> {
        if self.semaphore_type != vk::SemaphoreTypeKHR::TIMELINE {
            panic!("Semaphore type is not TIMELINE!");
        }

        let mut last_signal_value = self.last_signal_value.lock();
        *last_signal_value += 1;

        let signal_info = vk::SemaphoreSignalInfoKHR::builder()
            .semaphore(self.native)
            .value(*last_signal_value);

        unsafe { self.device_wrapper.ts_khr.signal_semaphore(&signal_info)? };

        Ok(*last_signal_value)
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        if self.semaphore_type == vk::SemaphoreTypeKHR::TIMELINE {
            self.wait(*self.last_signal_value.lock()).unwrap();
        }
        unsafe { self.device_wrapper.native.destroy_semaphore(self.native, None) };
    }
}