use crate::{Device, Format};
use ash::vk;
use common::types::HashMap;
use spirv_cross::spirv;
use std::sync::Arc;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ShaderStageFlags(pub(crate) vk::ShaderStageFlags);

impl ShaderStageFlags {
    pub const VERTEX: Self = Self(vk::ShaderStageFlags::VERTEX);
    pub const PIXEL: Self = Self(vk::ShaderStageFlags::FRAGMENT);
    pub const COMPUTE: Self = Self(vk::ShaderStageFlags::COMPUTE);
}
vk_bitflags_impl!(ShaderStageFlags, vk::ShaderStageFlags);

/// Single shader stage
pub type ShaderStage = ShaderStageFlags;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct VInputRate(pub(crate) vk::VertexInputRate);

impl VInputRate {
    pub const VERTEX: Self = Self(vk::VertexInputRate::VERTEX);
    pub const INSTANCE: Self = Self(vk::VertexInputRate::INSTANCE);
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BindingType(pub(crate) vk::DescriptorType);

impl BindingType {
    pub const UNIFORM_BUFFER: Self = Self(vk::DescriptorType::UNIFORM_BUFFER);
    pub const UNIFORM_BUFFER_DYNAMIC: Self = Self(vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC);
    pub const STORAGE_BUFFER: Self = Self(vk::DescriptorType::STORAGE_BUFFER);
    pub const SAMPLED_IMAGE: Self = Self(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);
    pub const STORAGE_IMAGE: Self = Self(vk::DescriptorType::STORAGE_IMAGE);
    pub const INPUT_ATTACHMENT: Self = Self(vk::DescriptorType::INPUT_ATTACHMENT);
}

pub type BindingId = u32;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
pub struct BindingLoc {
    pub descriptor_set: u32,
    pub id: BindingId,
}

impl BindingLoc {
    pub fn new(descriptor_set: u32, id: u32) -> Self {
        Self { descriptor_set, id }
    }
}

#[derive(Copy, Clone)]
pub struct ShaderBindingDescription {
    pub stage_flags: ShaderStageFlags,
    pub binding_type: BindingType,
    pub count: u32,
    pub readable: bool,
    pub writable: bool,
}

impl ShaderBindingDescription {
    pub fn pure_binding(&self) -> ShaderBinding {
        ShaderBinding {
            stage_flags: self.stage_flags,
            binding_type: self.binding_type,
            count: self.count,
        }
    }
}

#[derive(Copy, Clone)]
pub struct ShaderBinding {
    pub stage_flags: ShaderStageFlags,
    pub binding_type: BindingType,
    pub count: u32,
}

impl ShaderBinding {
    pub fn new(ty: BindingType) -> Self {
        Self {
            stage_flags: ShaderStageFlags::VERTEX | ShaderStageFlags::PIXEL | ShaderStageFlags::COMPUTE,
            binding_type: ty,
            count: 1,
        }
    }

    pub fn auto_describe(&self) -> ShaderBindingDescription {
        ShaderBindingDescription {
            stage_flags: self.stage_flags,
            binding_type: self.binding_type,
            count: self.count,
            readable: true,
            writable: matches!(
                self.binding_type,
                BindingType::STORAGE_BUFFER | BindingType::STORAGE_IMAGE
            ),
        }
    }
}

pub struct Shader {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::ShaderModule,
    pub(crate) stage: ShaderStage,
    // [location, format]
    pub(crate) vertex_location_inputs: HashMap<u32, (Format, VInputRate)>,
    pub(crate) named_bindings: HashMap<String, (BindingLoc, ShaderBindingDescription)>,
    pub(crate) _push_constants: HashMap<String, spirv::BufferRange>,
    pub(crate) push_constants_size: u32,
}

impl Shader {
    pub fn stage(&self) -> ShaderStage {
        self.stage
    }

    pub fn vertex_location_inputs(&self) -> &HashMap<u32, (Format, VInputRate)> {
        &self.vertex_location_inputs
    }

    pub fn named_bindings(&self) -> &HashMap<String, (BindingLoc, ShaderBindingDescription)> {
        &self.named_bindings
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        unsafe {
            self.device
                .wrapper
                .native
                .destroy_shader_module(self.native, None)
        };
    }
}
