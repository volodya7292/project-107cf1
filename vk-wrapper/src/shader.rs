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
vk_bitflags_impl!(ShaderStage, vk::ShaderStageFlags);

#[derive(Copy, Clone, PartialEq)]
pub struct ShaderBindingMod(u32);

impl ShaderBindingMod {
    pub const DEFAULT: Self = Self(0);
    pub const DYNAMIC_OFFSET: Self = Self(1);
    pub const DYNAMIC_UPDATE: Self = Self(2);
}

pub struct BindingType(pub(crate) vk::DescriptorType);

impl BindingType {
    pub const UNIFORM_BUFFER: Self = Self(vk::DescriptorType::UNIFORM_BUFFER);
    pub const UNIFORM_BUFFER_DYNAMIC: Self = Self(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC);
    pub const STORAGE_BUFFER: Self = Self(vk::DescriptorType::STORAGE_BUFFER);
    pub const SAMPLED_IMAGE: Self = Self(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
    pub const STORAGE_IMAGE: Self = Self(vk::DescriptorType::STORAGE_IMAGE);
}

pub struct ShaderBinding {
    pub binding_type: BindingType,
    pub descriptor_set: u32,
    pub id: u32,
    pub count: u32,
}

pub struct Shader {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::ShaderModule,
    pub(crate) stage: ShaderStage,
    pub(crate) input_locations: HashMap<u32, Format>,
    // [location, format]
    pub(crate) bindings: HashMap<String, ShaderBinding>,
    pub(crate) _push_constants: HashMap<String, spirv::BufferRange>,
    pub(crate) push_constants_size: u32,
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_shader_module(self.native, None) };
    }
}
