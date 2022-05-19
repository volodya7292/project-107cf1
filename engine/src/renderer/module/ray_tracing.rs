use crate::renderer::module::lbvh_generation::LBVHGenerationModule;
use crate::renderer::module::morton_bitonic_sort::MortonBitonicSort;
use crate::renderer::module::morton_codes_for_triangles::MortonCodesForTrianglesModule;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    DeviceError, Pipeline, PipelineSignature, PipelineStageFlags,
};

pub struct RayTracingModule {
    mct: MortonCodesForTrianglesModule,
    mbs: MortonBitonicSort,
    bvh_gen: LBVHGenerationModule,
}

impl RayTracingModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        Self {
            mct: MortonCodesForTrianglesModule::new(device, global_buffer),
            mbs: MortonBitonicSort::new(device, global_buffer),
            bvh_gen: LBVHGenerationModule::new(device, global_buffer),
        }
    }

    fn compute_bvh(&self) {}
}
