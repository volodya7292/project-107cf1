use crate::renderer::helpers::RendererError;
use crate::renderer::module::bottom_lbvh_generation::BottomLBVHGenModule;
use crate::renderer::module::bottom_lbvh_node_bounds::BottomLBVHNodeBoundsModule;
use crate::renderer::module::bottom_lbvh_prepare_leaves::{BLPLPayload, BottomLBVHPrepareLeavesModule};
use crate::renderer::module::morton_bitonic_sort::MortonBitonicSortModule;
use crate::renderer::module::top_lbvh_bounds::TopLBVHBoundsModule;
use crate::renderer::module::top_lbvh_generation::TopLBVHGenModule;
use crate::renderer::module::top_lbvh_leaf_bounds::TopLBVHLeafBoundsModule;
use crate::renderer::module::top_lbvh_node_bounds::TopLBVHNodeBoundsModule;
use crate::renderer::module::top_lbvh_prepare_leaves::TopLBVHPrepareLeavesModule;
use crate::renderer::GBVertexMesh;
use crate::utils::HashMap;
use nalgebra_glm::{Mat4, Vec3};
use parking_lot::Mutex;
use range_alloc::RangeAllocator;
use std::mem;
use std::sync::Arc;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    Pipeline, PipelineSignature, PipelineStageFlags, Queue, QueueType, SubmitPacket,
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
    device: Arc<Device>,

    bitonic_sort: MortonBitonicSortModule,

    bl_prepare_leaves: BottomLBVHPrepareLeavesModule,
    bl_gen: BottomLBVHGenModule,
    bl_node_bounds: BottomLBVHNodeBoundsModule,

    tl_leaf_bounds: TopLBVHLeafBoundsModule,
    tl_bounds: TopLBVHBoundsModule,
    tl_prepare_leaves: TopLBVHPrepareLeavesModule,
    tl_gen: TopLBVHGenModule,
    tl_node_bounds: TopLBVHNodeBoundsModule,
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
            device: Arc::clone(device),
            bl_prepare_leaves: BottomLBVHPrepareLeavesModule::new(device, global_buffer),
            bitonic_sort: MortonBitonicSortModule::new(device, global_buffer),
            bl_gen: BottomLBVHGenModule::new(device, global_buffer),
            bl_node_bounds: BottomLBVHNodeBoundsModule::new(device, global_buffer),
            tl_leaf_bounds: TopLBVHLeafBoundsModule::new(device, global_buffer),
            tl_bounds: TopLBVHBoundsModule::new(device, global_buffer),
            tl_prepare_leaves: TopLBVHPrepareLeavesModule::new(device, global_buffer),
            tl_gen: TopLBVHGenModule::new(device, global_buffer),
            tl_node_bounds: TopLBVHNodeBoundsModule::new(device, global_buffer),
        }
    }

    pub(crate) fn on_update(
        &self,
        staging_cl: Arc<Mutex<CmdList>>,
        staging_submit: &mut SubmitPacket,
        gb_allocator: &mut RangeAllocator<u32>,
        gvb_meshes: &HashMap<usize, GBVertexMesh>,
        dirty_meshes: &[usize],
    ) -> Result<(), RendererError> {
        let morton_codes_offset = gb_allocator.allocate_range(1024 * 1024 * mem::size_of::<u32>() as u32)?;
        let temp_bounds_offset = gb_allocator.allocate_range(1024 * 1024 * mem::size_of::<Bounds>() as u32);

        if temp_bounds_offset.is_err() {
            gb_allocator.free_range(morton_codes_offset);
            return Err(temp_bounds_offset.err().unwrap().into());
        }
        let temp_bounds_offset = temp_bounds_offset.unwrap();

        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        let mut cl = staging_cl.lock();
        cl.begin(true).unwrap();

        // TODO: implement construction for non-indexed meshes

        let payloads: Vec<_> = dirty_meshes
            .iter()
            .map(|mesh_ptr| {
                let gvb_mesh = &gvb_meshes[mesh_ptr];
                BLPLPayload {
                    indices_offset: gvb_mesh.gb_indices_offset,
                    vertices_offset: gvb_mesh.gb_binding_offsets[0],
                    morton_codes_offset: morton_codes_offset.start,
                    leaf_bounds_offset: temp_bounds_offset.start,
                    n_triangles: gvb_mesh.raw.index_count / 3,
                    mesh_bound_min: gvb_mesh.raw.aabb.0,
                    mesh_bound_max: gvb_mesh.raw.aabb.1,
                }
            })
            .collect();

        self.bl_prepare_leaves.dispatch(&mut cl, &payloads);

        cl.end().unwrap();
        drop(cl);
        unsafe { graphics_queue.submit(staging_submit).unwrap() };
        staging_submit.wait().unwrap();

        gb_allocator.free_range(temp_bounds_offset);
        gb_allocator.free_range(morton_codes_offset);

        Ok(())
    }
}
