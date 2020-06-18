use crate::device::DeviceWrapper;
use ash::version::DeviceV1_0;
use ash::version::DeviceV1_2;
use ash::vk;
use std::cell::Cell;
use std::sync::Arc;
use std::slice;

pub struct Semaphore {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) native: vk::Semaphore,
    pub(crate) semaphore_type: vk::SemaphoreType,
    pub(crate) last_signal_value: Cell<u64>,
}

impl Semaphore {
    pub fn wait(&self, value: u64) -> Result<(), vk::Result> {
        if self.semaphore_type != vk::SemaphoreType::TIMELINE {
            panic!("Semaphore type is not TIMELINE!");
        }

        let wait_info = vk::SemaphoreWaitInfo::builder()
            .semaphores(slice::from_ref(&self.native))
            .values(slice::from_ref(&value));
        unsafe {
            self.device_wrapper
                .0
                .wait_semaphores(self.device_wrapper.0.handle(), &wait_info, u64::MAX)
        }
    }

    pub fn signal(&self) -> Result<u64, vk::Result> {
        if self.semaphore_type != vk::SemaphoreType::TIMELINE {
            panic!("Semaphore type is not TIMELINE!");
        }

        self.last_signal_value.set(self.last_signal_value.get() + 1);

        let signal_info = vk::SemaphoreSignalInfo::builder()
            .semaphore(self.native)
            .value(self.last_signal_value.get());
        unsafe {
            self.device_wrapper
                .0
                .signal_semaphore(self.device_wrapper.0.handle(), &signal_info)?
        };

        Ok(self.last_signal_value.get())
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        if self.semaphore_type == vk::SemaphoreType::TIMELINE {
            self.wait(self.last_signal_value.get()).unwrap();
        }
        unsafe { self.device_wrapper.0.destroy_semaphore(self.native, None) };
    }
}
