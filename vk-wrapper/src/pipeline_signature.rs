use crate::{DescriptorPool, Device, Shader, ShaderStage};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

pub struct PipelineSignature {
    pub(crate) device: Arc<Device>,
    pub(crate) native: [vk::DescriptorSetLayout; 4],
    pub(crate) descriptor_sizes: [Vec<vk::DescriptorPoolSize>; 4],
    pub(crate) binding_types: HashMap<u32, vk::DescriptorType>,
    pub(crate) push_constant_ranges: HashMap<ShaderStage, (u32, u32)>,
    pub(crate) push_constants_size: u32,
    pub(crate) shaders: HashMap<ShaderStage, Arc<Shader>>,
}

impl PipelineSignature {
    pub fn create_pool(
        self: &Arc<Self>,
        set_id: u32,
        max_inputs: u32,
    ) -> Result<Arc<DescriptorPool>, vk::Result> {
        let mut pool_sizes = self.descriptor_sizes[set_id as usize].clone();
        for mut pool_size in &mut pool_sizes {
            pool_size.descriptor_count *= max_inputs;
        }

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(max_inputs)
            .pool_sizes(&pool_sizes);

        Ok(Arc::new(DescriptorPool {
            device: Arc::clone(&self.device),
            signature: Arc::clone(&self),
            descriptor_set: set_id,
            native: unsafe { self.device.wrapper.0.create_descriptor_pool(&pool_info, None)? },
            free_sets: Mutex::new(Vec::with_capacity(65535)),
        }))
    }
}

impl Drop for PipelineSignature {
    fn drop(&mut self) {
        for native_set_layout in &self.native {
            if *native_set_layout != vk::DescriptorSetLayout::default() {
                unsafe {
                    self.device
                        .wrapper
                        .0
                        .destroy_descriptor_set_layout(*native_set_layout, None)
                };
            }
        }
    }
}
