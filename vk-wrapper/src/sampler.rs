use crate::Device;
use ash::vk;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub struct SamplerReduction(pub(crate) vk::SamplerReductionMode);

pub struct Sampler {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::Sampler,
}

impl Sampler {
    pub const REDUCTION_MIN: SamplerReduction = SamplerReduction(vk::SamplerReductionMode::MIN);
    pub const REDUCTION_MAX: SamplerReduction = SamplerReduction(vk::SamplerReductionMode::MAX);
    pub const REDUCTION_AVG: SamplerReduction = SamplerReduction(vk::SamplerReductionMode::WEIGHTED_AVERAGE);
}

impl PartialEq for Sampler {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Eq for Sampler {}

impl Hash for Sampler {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.native.hash(state);
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.native.destroy_sampler(self.native, None) };
    }
}
