#[cfg(target_os = "macos")]
use crate::platform::metal;
use crate::{DeviceError, Instance};
use ash::vk;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use std::sync::Arc;

pub struct Surface {
    pub(crate) instance: Arc<Instance>,
    pub(crate) native: vk::SurfaceKHR,
}

impl Surface {
    /// Updates surface backing properties such as scale factor if necessary.
    /// # Safety
    pub fn update(window: impl HasWindowHandle) -> Result<(), DeviceError> {
        match window.window_handle().unwrap().as_raw() {
            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(handle) => unsafe { metal::metal_layer_update(handle) },
            _ => {}
        }
        Ok(())
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance.surface_khr.destroy_surface(self.native, None);
        }
    }
}
