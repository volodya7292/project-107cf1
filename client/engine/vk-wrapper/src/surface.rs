use std::sync::Arc;

use ash::vk;

use crate::Instance;

pub struct Surface {
    pub(crate) instance: Arc<Instance>,
    pub(crate) native: vk::SurfaceKHR,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance.surface_khr.destroy_surface(self.native, None);
        }
    }
}
