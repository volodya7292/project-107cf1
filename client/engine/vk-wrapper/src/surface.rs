use crate::platform::metal;
use crate::{DeviceError, Instance};
use ash::vk;
use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use std::sync::Arc;

pub struct Surface {
    pub(crate) instance: Arc<Instance>,
    pub(crate) native: vk::SurfaceKHR,
}

impl Surface {
    /// Updates surface backing properties such as scale factor if necessary.
    pub unsafe fn update(window: impl HasRawWindowHandle) -> Result<(), DeviceError> {
        match window.raw_window_handle() {
            #[cfg(any(target_os = "macos"))]
            RawWindowHandle::AppKit(handle) => metal::metal_layer_update(handle),
            _ => return Err(DeviceError::VkError(vk::Result::ERROR_EXTENSION_NOT_PRESENT)), // not supported
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
