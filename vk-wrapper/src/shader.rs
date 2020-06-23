use crate::{Device, Format};
use ash::version::DeviceV1_0;
use ash::vk;
use spirv_cross::spirv;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ShaderStage(pub(crate) vk::ShaderStageFlags);

impl ShaderStage {
    pub const VERTEX: Self = Self(vk::ShaderStageFlags::VERTEX);
    pub const PIXEL: Self = Self(vk::ShaderStageFlags::FRAGMENT);
    pub const GEOMETRY: Self = Self(vk::ShaderStageFlags::GEOMETRY);
    pub const COMPUTE: Self = Self(vk::ShaderStageFlags::COMPUTE);
}

#[derive(Copy, Clone, PartialEq)]
pub struct ShaderBindingMod(u32);

impl ShaderBindingMod {
    pub const DYNAMIC_UPDATE: Self = Self(0);
}

pub(crate) struct ShaderBinding {
    pub(crate) binding_type: vk::DescriptorType,
    pub(crate) id: u32,
    pub(crate) count: u32,
}

pub struct Shader {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::ShaderModule,
    pub(crate) stage: ShaderStage,
    pub(crate) input_locations: HashMap<u32, Format>,
    // [location, format]
    pub(crate) bindings: HashMap<String, ShaderBinding>,
    pub(crate) push_constants: HashMap<String, spirv::BufferRange>,
    pub(crate) push_constants_size: u32,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_shader_module(self.native, None) };
    }
}
