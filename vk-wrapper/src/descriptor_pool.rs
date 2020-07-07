use crate::{Device, PipelineInput, PipelineSignature};
use ash::version::DeviceV1_0;
use ash::vk;
use std::slice;
use std::sync::{Arc, Mutex};

pub struct DescriptorPool {
    pub(crate) device: Arc<Device>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) descriptor_set: u32,
    pub(crate) native: vk::DescriptorPool,
    pub(crate) free_sets: Mutex<Vec<vk::DescriptorSet>>,
}

impl DescriptorPool {
    pub fn allocate_input(self: &Arc<Self>) -> Result<Arc<PipelineInput>, vk::Result> {
        let mut free_sets = self.free_sets.lock().unwrap();

        let native_set = if free_sets.is_empty() {
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.native)
                .set_layouts(slice::from_ref(
                    &self.signature.native[self.descriptor_set as usize],
                ));

            unsafe { self.device.wrapper.0.allocate_descriptor_sets(&alloc_info)?[0] }
        } else {
            let index = free_sets.len() - 1;
            free_sets.remove(index)
        };

        Ok(Arc::new(PipelineInput {
            pool: Arc::clone(self),
            native: native_set,
            used_buffers: Default::default(),
            used_images: Default::default(),
        }))
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_descriptor_pool(self.native, None) };
    }
}
