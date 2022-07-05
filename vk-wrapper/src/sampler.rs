use crate::DeviceWrapper;
use ash::vk;
use std::sync::Arc;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct SamplerFilter(pub(crate) vk::Filter);

impl SamplerFilter {
    pub const NEAREST: Self = Self(vk::Filter::NEAREST);
    pub const LINEAR: Self = Self(vk::Filter::LINEAR);
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct SamplerMipmap(pub(crate) vk::SamplerMipmapMode);

impl SamplerMipmap {
    pub const NEAREST: Self = Self(vk::SamplerMipmapMode::NEAREST);
    pub const LINEAR: Self = Self(vk::SamplerMipmapMode::LINEAR);
}

pub struct Sampler {
    pub(crate) device_wrapper: Arc<DeviceWrapper>,
    pub(crate) native: vk::Sampler,
    pub(crate) min_filter: SamplerFilter,
    pub(crate) mag_filter: SamplerFilter,
    pub(crate) mipmap: SamplerMipmap,
}

impl Sampler {}

impl PartialEq for Sampler {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device_wrapper.native.destroy_sampler(self.native, None) };
    }
}
