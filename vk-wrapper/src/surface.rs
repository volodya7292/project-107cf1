use crate::Instance;
use ash::vk;
use std::rc::Rc;
use std::sync::Arc;

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
