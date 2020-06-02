use crate::Instance;
use ash::vk;
use std::rc::Rc;

pub struct Surface {
    pub(crate) instance: Rc<Instance>,
    pub(crate) native: vk::SurfaceKHR,
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance.surface_khr.destroy_surface(self.native, None);
        }
    }
}
