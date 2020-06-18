use crate::{Device, ShaderStage};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;

pub struct PipelineSignature {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::DescriptorSetLayout,
    pub(crate) descriptor_sizes: Vec<vk::DescriptorPoolSize>,
    pub(crate) descriptor_sizes_indices: HashMap<vk::DescriptorType, u32>,
    pub(crate) binding_types: HashMap<u32, vk::DescriptorType>,
    pub(crate) push_constant_ranges: HashMap<ShaderStage, (u32, u32)>,
    pub(crate) push_constants_size: u32,
}

impl Drop for PipelineSignature {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wrapper
                .0
                .destroy_descriptor_set_layout(self.native, None)
        };
    }
}
