use crate::Device;
use ash::vk;
use std::sync::Arc;

pub struct QueryPool {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::QueryPool,
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.native.destroy_query_pool(self.native, None) };
    }
}
