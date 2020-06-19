use crate::{Device, PipelineInput, ShaderStage};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::{slice, sync::Arc};

pub struct PipelineSignature {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::DescriptorSetLayout,
    pub(crate) descriptor_sizes: Vec<vk::DescriptorPoolSize>,
    pub(crate) descriptor_sizes_indices: HashMap<vk::DescriptorType, u32>,
    pub(crate) binding_types: HashMap<u32, vk::DescriptorType>,
    pub(crate) push_constant_ranges: HashMap<ShaderStage, (u32, u32)>,
    pub(crate) push_constants_size: u32,
}

impl PipelineSignature {
    pub fn create_input(self: &Arc<Self>) -> Result<Arc<PipelineInput>, vk::Result> {
        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
            .max_sets(1)
            .pool_sizes(&self.descriptor_sizes);
        let pool = unsafe { self.device.wrapper.0.create_descriptor_pool(&pool_info, None)? };

        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(pool)
            .set_layouts(slice::from_ref(&self.native));
        let descriptor_set = unsafe { self.device.wrapper.0.allocate_descriptor_sets(&alloc_info)?[0] };

        Ok(Arc::new(PipelineInput {
            device: Arc::clone(&self.device),
            signature: Arc::clone(self),
            pool,
            native: descriptor_set,
            used_buffers: HashMap::new(),
            used_images: HashMap::new(),
        }))
    }
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
