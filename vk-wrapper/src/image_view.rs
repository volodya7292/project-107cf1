use crate::ImageWrapper;
use ash::version::DeviceV1_0;
use ash::vk;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub struct ImageView {
    pub(crate) image_wrapper: Arc<ImageWrapper>,
    pub(crate) native: vk::ImageView,
}

impl PartialEq for ImageView {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Eq for ImageView {}

impl Hash for ImageView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.native.hash(state);
    }
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
