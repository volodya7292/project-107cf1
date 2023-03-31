use std::sync::Arc;

use ash::vk;

use crate::ImageWrapper;

pub struct ImageView {
    pub(crate) image_wrapper: Arc<ImageWrapper>,
    pub(crate) native: vk::ImageView,
}

impl PartialEq for ImageView {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.image_wrapper
                .device
                .wrapper
                .native
                .destroy_image_view(self.native, None)
        };
    }
}
