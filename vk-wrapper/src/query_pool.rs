use crate::Device;
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::Arc;

pub struct QueryPool {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::QueryPool,
}

impl Drop for QueryPool {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_query_pool(self.native, None) };
    }
}
