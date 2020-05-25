use crate::Device;
use ash::version::DeviceV1_0;
use ash::vk;
use spirv_cross::spirv;
use std::collections::HashMap;
use std::rc::Rc;

pub struct ShaderStage(pub(crate) vk::ShaderStageFlags);

impl ShaderStage {
    pub const VERTEX: Self = Self(vk::ShaderStageFlags::VERTEX);
    pub const PIXEL: Self = Self(vk::ShaderStageFlags::FRAGMENT);
    pub const GEOMETRY: Self = Self(vk::ShaderStageFlags::GEOMETRY);
    pub const COMPUTE: Self = Self(vk::ShaderStageFlags::COMPUTE);
}

#[derive(Copy, Clone, PartialEq)]
pub struct ShaderBindingType(u32);

impl ShaderBindingType {
    pub const DEFAULT: Self = Self(0);
    pub const DYNAMIC_RANGE: Self = Self(1);
    pub const DYNAMIC_UPDATE: Self = Self(2);
}

pub(crate) struct ShaderBinding {
    pub(crate) binding_type: vk::DescriptorType,
    pub(crate) id: u32,
    pub(crate) count: u32,
}

pub struct Shader {
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::ShaderModule,
    pub(crate) stage: ShaderStage,
    pub(crate) bindings: HashMap<String, ShaderBinding>,
    pub(crate) push_constants: HashMap<String, spirv::BufferRange>,
    pub(crate) push_constants_size: u32,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.native.destroy_shader_module(self.native, None) };
    }
}
