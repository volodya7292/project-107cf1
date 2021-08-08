use crate::{DescriptorPool, Device, Shader, ShaderStage};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub struct PipelineSignature {
    pub(crate) device: Arc<Device>,
    pub(crate) native: [vk::DescriptorSetLayout; 4],
    pub(crate) pipeline_layout: vk::PipelineLayout,
    pub(crate) descriptor_sizes: [Vec<vk::DescriptorPoolSize>; 4],
    pub(crate) binding_types: [HashMap<u32, vk::DescriptorType>; 4],
    pub(crate) _push_constants_size: u32,
    pub(crate) shaders: HashMap<ShaderStage, Arc<Shader>>,
}

impl PipelineSignature {
    pub fn create_pool(
        self: &Arc<Self>,
        set_layout_id: u32,
        base_reserve: u32,
    ) -> Result<DescriptorPool, vk::Result> {
        let mut pool = DescriptorPool {
            device: Arc::clone(&self.device),
            signature: Arc::clone(&self),
            set_layout_id,
            native: Default::default(),
            allocated: vec![],
            free_sets: Default::default(),
        };
        pool.alloc_next_pool(base_reserve.next_power_of_two())?;
        Ok(pool)
    }
}

impl PartialEq for PipelineSignature {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Eq for PipelineSignature {}

impl Hash for PipelineSignature {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.native.hash(state);
    }
}

impl Drop for PipelineSignature {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wrapper
                .native
                .destroy_pipeline_layout(self.pipeline_layout, None)
        };

        for native_set_layout in &self.native {
            if *native_set_layout != vk::DescriptorSetLayout::default() {
                unsafe {
                    self.device
                        .wrapper
                        .native
                        .destroy_descriptor_set_layout(*native_set_layout, None)
                };
            }
        }
    }
}
