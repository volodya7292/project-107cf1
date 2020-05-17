use crate::{adapter::Adapter, Instance};
use ash::vk;
use std::rc::Rc;

pub struct Surface {
    pub(crate) instance: Rc<Instance>,
    pub(crate) native: vk::SurfaceKHR,
}

impl Surface {
    pub(crate) fn get_physical_device_surface_support(&self, adapter: &Adapter, queue_family_index: u32) -> Result<bool, vk::Result> {
        unsafe {
            self.instance.surface_khr
                .get_physical_device_surface_support(adapter.native, queue_family_index, self.native)
        }
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.instance.surface_khr.destroy_surface(self.native, None);
        }
    }
}
