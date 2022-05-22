use crate::renderer::module::bounds_for_bottom_lbvh::BoundsForBottomLBVHModule;
use crate::renderer::module::lbvh_generation::LBVHGenerationModule;
use crate::renderer::module::leaf_aabbs_for_lbvhs::LeafAABBsForLBVHsModule;
use crate::renderer::module::morton_bitonic_sort::MortonBitonicSort;
use crate::renderer::module::prepare_bottom_lbvh_leaves::PrepareBottomLBVHLeavesModule;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    DeviceError, Pipeline, PipelineSignature, PipelineStageFlags,
};

pub struct RayTracingModule {
    pt: PrepareBottomLBVHLeavesModule,
    mbs: MortonBitonicSort,
    bvh_gen: LBVHGenerationModule,
    bounds_bottom_lbvh: BoundsForBottomLBVHModule,
    aabbs_for_lbvhs: LeafAABBsForLBVHsModule,
}

impl RayTracingModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        Self {
            pt: PrepareBottomLBVHLeavesModule::new(device, global_buffer),
            mbs: MortonBitonicSort::new(device, global_buffer),
            bvh_gen: LBVHGenerationModule::new(device, global_buffer),
            bounds_bottom_lbvh: BoundsForBottomLBVHModule::new(device, global_buffer),
            aabbs_for_lbvhs: LeafAABBsForLBVHsModule::new(device, global_buffer),
        }
    }

    fn compute_bvh(&self) {}
}
