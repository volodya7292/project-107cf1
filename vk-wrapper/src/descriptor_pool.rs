use crate::{Device, DeviceBuffer, Image, PipelineInput, PipelineSignature};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::slice;
use std::sync::{Arc, Mutex};

pub struct DescriptorPool {
    pub(crate) device: Arc<Device>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) descriptor_set: u32,
    pub(crate) native: vk::DescriptorPool,
    pub(crate) inputs: Mutex<Vec<Option<Arc<PipelineInput>>>>,

    // (u32, u32, u32) : (descriptor index, binding id, array index)
    pub(crate) used_buffers: Mutex<HashMap<(u32, u32, u32), Arc<DeviceBuffer>>>,
    pub(crate) used_images: Mutex<HashMap<(u32, u32, u32), Arc<Image>>>,
}

impl DescriptorPool {
    pub fn get_input(self: &Arc<Self>, index: u32) -> Result<Arc<PipelineInput>, vk::Result> {
        let mut inputs = self.inputs.lock().unwrap();
        let input = &mut inputs[index as usize];

        if input.is_none() {
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.native)
                .set_layouts(slice::from_ref(
                    &self.signature.native[self.descriptor_set as usize],
                ));

            *input = Some(Arc::new(PipelineInput {
                pool: Arc::clone(self),
                native: unsafe { self.device.wrapper.0.allocate_descriptor_sets(&alloc_info)?[0] },
            }))
        }

        Ok(input.as_ref().unwrap().clone())
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_descriptor_pool(self.native, None) };
    }
}
