use crate::ImageWrapper;
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::Arc;

pub struct ImageView {
    pub(crate) image_wrapper: Arc<ImageWrapper>,
    pub(crate) native: vk::ImageView,
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.image_wrapper
                .device
                .wrapper
                .0
                .destroy_image_view(self.native, None)
        };
    }
}
