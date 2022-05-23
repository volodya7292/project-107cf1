use crate::renderer::module::bottom_lbvh_generation::BottomLBVHGenModule;
use crate::renderer::module::bottom_lbvh_node_bounds::BottomLBVHNodeBoundsModule;
use crate::renderer::module::bottom_lbvh_prepare_leaves::BottomLBVHPrepareLeavesModule;
use crate::renderer::module::morton_bitonic_sort::MortonBitonicSortModule;
use crate::renderer::module::top_lbvh_bounds::TopLBVHBoundsModule;
use crate::renderer::module::top_lbvh_generation::TopLBVHGenModule;
use crate::renderer::module::top_lbvh_leaf_bounds::TopLBVHLeafBoundsModule;
use crate::renderer::module::top_lbvh_prepare_leaves::TopLBVHPrepareLeavesModule;
use nalgebra_glm::{Mat4, Vec3};
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    DeviceError, Pipeline, PipelineSignature, PipelineStageFlags,
};

// Scene acceleration structure construction pipeline:
//
// 1. Build Bottom LBVHs.
//    1. Prepare leaves (compute bounds and morton code for each triangle).
//    2. Sort morton codes.
//    3. Generate hierarchy.
//    4. Compute bounds for each node.
//
// 2. Build Top (scene) LBVH.
//    1. Calculate bounds for each leaf node.
//    2. Calculate overall (scene) bounding box.
//    3. Prepare leaves (compute morton code for each leaf).
//    4. Sort morton codes.
//    5. Generate hierarchy.
//    6. Compute bounds for each node.

pub struct RayTracingModule {
    mbs: MortonBitonicSortModule,

    pbll: BottomLBVHPrepareLeavesModule,
    blg: BottomLBVHGenModule,
    blnb: BottomLBVHNodeBoundsModule,

    tllb: TopLBVHLeafBoundsModule,
    tlb: TopLBVHBoundsModule,
    tlpl: TopLBVHPrepareLeavesModule,
    tlg: TopLBVHGenModule,
}

#[repr(C)]
struct Bounds {
    p_min: Vec3,
    p_max: Vec3,
}

#[repr(C)]
pub struct LBVHNode {
    bounds: Bounds,
    element_id: u32,
    parent: u32,
    child_a: u32,
    child_b: u32,
}

#[repr(C)]
struct LBVHInstance {
    indices_offset: u32,
    vertices_offset: u32,
    nodes_offset: u32,
    transform: Mat4,
    transform_inverse: Mat4,
    bounds: Bounds,
}

#[repr(C)]
struct TopLBVHNode {
    instance: LBVHInstance,
    parent: u32,
    child_a: u32,
    child_b: u32,
}

impl RayTracingModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        Self {
            pbll: BottomLBVHPrepareLeavesModule::new(device, global_buffer),
            mbs: MortonBitonicSortModule::new(device, global_buffer),
            blg: BottomLBVHGenModule::new(device, global_buffer),
            blnb: BottomLBVHNodeBoundsModule::new(device, global_buffer),
            tllb: TopLBVHLeafBoundsModule::new(device, global_buffer),
            tlb: TopLBVHBoundsModule::new(device, global_buffer),
            tlpl: TopLBVHPrepareLeavesModule::new(device, global_buffer),
            tlg: TopLBVHGenModule::new(device, global_buffer),
        }
    }

    fn compute_bvh(&self) {}
}
