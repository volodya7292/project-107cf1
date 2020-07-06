use crate::device::DeviceWrapper;
use ash::version::DeviceV1_0;
use ash::version::DeviceV1_2;
use ash::vk;
use std::slice;
use std::sync::{Arc, Mutex};

pub struct Semaphore {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) native: vk::Semaphore,
    pub(crate) semaphore_type: vk::SemaphoreType,
    pub(crate) last_signal_value: Mutex<u64>,
}

impl Semaphore {
    pub fn wait(&self, value: u64) -> Result<(), vk::Result> {
        if self.semaphore_type != vk::SemaphoreType::TIMELINE {
            panic!("Semaphore type is not TIMELINE!");
        }

        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(slice::from_ref(&self.native))
            .values(slice::from_ref(&value));
        unsafe { self.device_wrapper.0.wait_semaphores(&wait_info, u64::MAX) }
    }

    pub fn signal(&self) -> Result<u64, vk::Result> {
        if self.semaphore_type != vk::SemaphoreType::TIMELINE {
            panic!("Semaphore type is not TIMELINE!");
        }

        let mut last_signal_value = self.last_signal_value.lock().unwrap();
        *last_signal_value += 1;

        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.native)
            .value(*last_signal_value);
        unsafe { self.device_wrapper.0.signal_semaphore(&signal_info)? };

        Ok(*last_signal_value)
    }

    pub fn last_signal_value(&self) -> u64 {
        *self.last_signal_value.lock().unwrap()
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        if self.semaphore_type == vk::SemaphoreType::TIMELINE {
            self.wait(*self.last_signal_value.lock().unwrap()).unwrap();
        }
        unsafe { self.device_wrapper.0.destroy_semaphore(self.native, None) };
    }
}
