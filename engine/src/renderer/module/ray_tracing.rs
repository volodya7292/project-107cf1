use crate::renderer::module::aabbs_for_lbvhs::AABBsForLBVHsModule;
use crate::renderer::module::bounds_for_bottom_lbvh::BoundsForBottomLBVHModule;
use crate::renderer::module::lbvh_generation::LBVHGenerationModule;
use crate::renderer::module::morton_bitonic_sort::MortonBitonicSort;
use crate::renderer::module::prepare_triangles::PrepareTrianglesModule;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    DeviceError, Pipeline, PipelineSignature, PipelineStageFlags,
};

pub struct RayTracingModule {
    pt: PrepareTrianglesModule,
    mbs: MortonBitonicSort,
    bvh_gen: LBVHGenerationModule,
    bounds_bottom_lbvh: BoundsForBottomLBVHModule,
    aabbs_for_lbvhs: AABBsForLBVHsModule,
}

impl RayTracingModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        Self {
            pt: PrepareTrianglesModule::new(device, global_buffer),
            mbs: MortonBitonicSort::new(device, global_buffer),
            bvh_gen: LBVHGenerationModule::new(device, global_buffer),
            bounds_bottom_lbvh: BoundsForBottomLBVHModule::new(device, global_buffer),
            aabbs_for_lbvhs: AABBsForLBVHsModule::new(device, global_buffer),
        }
    }

    fn compute_bvh(&self) {}
}
