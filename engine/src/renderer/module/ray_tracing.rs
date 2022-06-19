use crate::ecs::component::internal::GlobalTransform;
use crate::ecs::scene::{Entity, Scene};
use crate::ecs::scene_storage::ComponentStorageImpl;
use crate::renderer::helpers::{LargeBuffer, LargeBufferAllocation, RendererError};
use crate::renderer::module::bottom_lbvh_generation::{BLGPayload, BottomLBVHGenModule};
use crate::renderer::module::bottom_lbvh_node_bounds::{BLNBPayload, BottomLBVHNodeBoundsModule};
use crate::renderer::module::bottom_lbvh_prepare_leaves::{BLPLPayload, BottomLBVHPrepareLeavesModule};
use crate::renderer::module::morton_bitonic_sort::{MBSPayload, MortonBitonicSortModule};
use crate::renderer::module::ray_tracer::{RTPayload, RayTracerModule};
use crate::renderer::module::top_lbvh_bounds::{TLBPayload, TopLBVHBoundsModule};
use crate::renderer::module::top_lbvh_generation::{TLGPayload, TopLBVHGenModule};
use crate::renderer::module::top_lbvh_leaf_bounds::{TLLBPayload, TopLBVHLeafBoundsModule};
use crate::renderer::module::top_lbvh_node_bounds::{TLNBPayload, TopLBVHNodeBoundsModule};
use crate::renderer::module::top_lbvh_prepare_leaves::{TLPLPayload, TopLBVHPrepareLeavesModule};
use crate::renderer::GBVertexMesh;
use crate::utils::{slice_as_bytes, HashMap};
use nalgebra_glm::{Mat4, Vec3};
use parking_lot::Mutex;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper::{
    AccessFlags, CmdList, Device, DeviceBuffer, HostBuffer, Image, PipelineStageFlags, Queue, SubmitPacket,
};

// Scene acceleration structure construction pipeline:
//
// 1. Build Bottom LBVHs.
//    1. Prepare leaves (compute morton code for each triangle).
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

const BLBVH_MAX_TRIANGLES: u32 = 1024 * 1024;
const SCENE_MAX_INSTANCES: u32 = 32768;

pub struct RayTracingModule {
    device: Arc<Device>,
    scene_lbvh_nodes_range: LargeBufferAllocation,

    bitonic_sort: MortonBitonicSortModule,

    bl_prepare_leaves: BottomLBVHPrepareLeavesModule,
    bl_gen: BottomLBVHGenModule,
    bl_node_bounds: BottomLBVHNodeBoundsModule,

    tl_leaf_bounds: TopLBVHLeafBoundsModule,
    tl_bounds: TopLBVHBoundsModule,
    tl_prepare_leaves: TopLBVHPrepareLeavesModule,
    tl_gen: TopLBVHGenModule,
    tl_node_bounds: TopLBVHNodeBoundsModule,

    ray_tracer: RayTracerModule,
}

#[derive(Debug)]
#[repr(C)]
struct MortonCode {
    code: u32,
    elem: u32,
}

#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
pub(crate) struct Bounds {
    p_min: Vec3,
    p_max: Vec3,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
pub struct LBVHNode {
    bounds: Bounds,
    element_id: u32,
    parent: u32,
    child_a: u32,
    child_b: u32,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct LBVHInstance {
    indices_offset: u32,
    vertices_offset: u32,
    nodes_offset: u32,
    transform: Mat4,
    transform_inverse: Mat4,
    bounds: Bounds,
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
struct TopLBVHNode {
    instance: LBVHInstance,
    parent: u32,
    child_a: u32,
    child_b: u32,
}

impl RayTracingModule {
    pub fn new(device: &Arc<Device>, global_buffer: &mut LargeBuffer) -> Self {
        let scene_lbvh_nodes_range = global_buffer
            .allocate((SCENE_MAX_INSTANCES * 2 - 1) * mem::size_of::<TopLBVHNode>() as u32)
            .unwrap();

        Self {
            device: Arc::clone(device),
            scene_lbvh_nodes_range,
            bl_prepare_leaves: BottomLBVHPrepareLeavesModule::new(device, global_buffer),
            bitonic_sort: MortonBitonicSortModule::new(device, global_buffer),
            bl_gen: BottomLBVHGenModule::new(device, global_buffer),
            bl_node_bounds: BottomLBVHNodeBoundsModule::new(device, global_buffer),
            tl_leaf_bounds: TopLBVHLeafBoundsModule::new(device, global_buffer),
            tl_bounds: TopLBVHBoundsModule::new(device, global_buffer),
            tl_prepare_leaves: TopLBVHPrepareLeavesModule::new(device, global_buffer),
            tl_gen: TopLBVHGenModule::new(device, global_buffer),
            tl_node_bounds: TopLBVHNodeBoundsModule::new(device, global_buffer),
            ray_tracer: RayTracerModule::new(device, global_buffer),
        }
    }

    pub(crate) fn on_update(
        &self,
        staging_cl: Arc<Mutex<CmdList>>,
        staging_submit: &mut SubmitPacket,
        global_buffer: &mut LargeBuffer,
        staging_buffer: &mut HostBuffer<u8>,
        gvb_meshes: &HashMap<usize, GBVertexMesh>,
        dirty_meshes: &[usize],
        entity_vertex_meshes: &HashMap<Entity, usize>,
        scene: &Scene,
    ) -> Result<(), RendererError> {
        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        let gb_barrier = global_buffer.barrier();
        let mut n_instances = entity_vertex_meshes.len() as u32;

        if n_instances > SCENE_MAX_INSTANCES {
            eprintln!("Max instances of BLBVH is reached! n_instances = {}", n_instances);
            n_instances = SCENE_MAX_INSTANCES;
        }

        let allocs = global_buffer.allocate_multiple(&[
            BLBVH_MAX_TRIANGLES * mem::size_of::<MortonCode>() as u32,
            BLBVH_MAX_TRIANGLES * mem::size_of::<u32>() as u32,
            BLBVH_MAX_TRIANGLES * mem::size_of::<Bounds>() as u32,
            n_instances * mem::size_of::<LBVHInstance>() as u32,
        ])?;
        let morton_codes_range = &allocs[0];
        let atomics_range = &allocs[1];
        let temp_bounds_range = &allocs[2];
        let instances_range = &allocs[3];

        // TODO: implement construction for non-indexed meshes

        // TODO: remove all debug barriers (used here to fix false-positive from vk sync check layer)

        let mut cl = staging_cl.lock();
        cl.begin(true).unwrap();

        // Build Bottom-level (mesh) LBVHs
        // -------------------------------------------------------------------------------------------------------------
        for mesh_ptr in dirty_meshes {
            let gvb_mesh = &gvb_meshes[mesh_ptr];
            let n_triangles = gvb_mesh.raw.index_count / 3;
            let n_nodes = n_triangles * 2 - 1;

            let nodes_range_len = n_nodes as u64 * mem::size_of::<LBVHNode>() as u64;
            let morton_codes_len = n_triangles as u64 * mem::size_of::<MortonCode>() as u64;
            let atomics_len = (n_triangles as u64 - 1) * mem::size_of::<u32>() as u64;

            if n_triangles > BLBVH_MAX_TRIANGLES {
                eprintln!(
                    "Max triangles for BLBVH is reached! Not processing this mesh. n_triangles = {}",
                    n_triangles
                );
                continue;
            }

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(morton_codes_range.start() as u64)
                    .size(morton_codes_len)
                    .src_access_mask(AccessFlags::SHADER_READ)
                    .dst_access_mask(AccessFlags::SHADER_WRITE)],
            );

            // 1. Prepare leaves
            let blpl_payload = BLPLPayload {
                indices_offset: gvb_mesh.gb_indices_offset,
                vertices_offset: gvb_mesh.gb_binding_offsets[0],
                morton_codes_offset: morton_codes_range.start(),
                nodes_offset: gvb_mesh.gb_rt_nodes_offset,
                leaves_bounds_offset: temp_bounds_range.start(),
                n_triangles,
                mesh_bound_min: gvb_mesh.raw.aabb.0,
                mesh_bound_max: gvb_mesh.raw.aabb.1,
            };
            self.bl_prepare_leaves.dispatch(&mut cl, &blpl_payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[
                    gb_barrier
                        .offset(morton_codes_range.start() as u64)
                        .size(morton_codes_len)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE),
                    gb_barrier
                        .offset(gvb_mesh.gb_rt_nodes_offset as u64)
                        .size(nodes_range_len)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::SHADER_READ),
                ],
            );

            // 2. Sort morton codes
            let blsmc_payload = MBSPayload {
                morton_codes_offset: morton_codes_range.start(),
                n_codes: n_triangles,
            };
            self.bitonic_sort.dispatch(&mut cl, &blsmc_payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(morton_codes_range.start() as u64)
                    .size(morton_codes_len)
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)],
            );

            // 3. Generate LBVH
            let blg_payload = BLGPayload {
                morton_codes_offset: morton_codes_range.start(),
                leaves_bounds_offset: temp_bounds_range.start(),
                nodes_offset: gvb_mesh.gb_rt_nodes_offset,
                n_triangles,
            };
            self.bl_gen.dispatch(&mut cl, &blg_payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE | PipelineStageFlags::TRANSFER,
                &[
                    gb_barrier
                        .offset(gvb_mesh.gb_rt_nodes_offset as u64)
                        .size(nodes_range_len)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE),
                    gb_barrier
                        .offset(atomics_range.start() as u64)
                        .size(atomics_len)
                        .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::TRANSFER_WRITE),
                ],
            );

            cl.fill_buffer2(global_buffer, atomics_range.start() as u64, atomics_len, 0);

            cl.barrier_buffer(
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(atomics_range.start() as u64)
                    .size(atomics_len)
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)],
            );

            // 4. Compute node bounds
            let blnb_payload = BLNBPayload {
                nodes_offset: gvb_mesh.gb_rt_nodes_offset,
                atomic_counters_offset: atomics_range.start(),
                n_triangles,
            };
            cl.debug_full_memory_barrier();
            self.bl_node_bounds.dispatch(&mut cl, &blnb_payload);
        }

        // Build Top-level (scene) LBVH
        // -------------------------------------------------------------------------------------------------------------
        let morton_codes_len = n_instances as u64 * mem::size_of::<MortonCode>() as u64;
        let atomics_len = n_instances as u64 * mem::size_of::<u32>() as u64;

        // Build instance buffer and upload it to GPU
        let g_transforms = scene.storage_read::<GlobalTransform>();
        let instances: Vec<_> = entity_vertex_meshes
            .iter()
            .map(|(entity, mesh_ptr)| {
                let mesh = &gvb_meshes[mesh_ptr];
                let g_trans = g_transforms.get(*entity).unwrap();

                LBVHInstance {
                    indices_offset: mesh.gb_indices_offset,
                    vertices_offset: mesh.gb_position_binding_offset,
                    nodes_offset: mesh.gb_rt_nodes_offset,
                    transform: g_trans.matrix,
                    transform_inverse: g_trans.matrix_inverse,
                    bounds: Bounds {
                        p_min: mesh.raw.aabb.0,
                        p_max: mesh.raw.aabb.1,
                    },
                }
            })
            .collect();

        let instances_bytes = unsafe { slice_as_bytes(&instances) };
        staging_buffer.write(0, instances_bytes);

        if instances.is_empty() {
            unsafe {
                staging_buffer.write_bytes(
                    0,
                    TopLBVHNode {
                        instance: LBVHInstance {
                            indices_offset: 0,
                            vertices_offset: 0,
                            nodes_offset: u32::MAX,
                            transform: Default::default(),
                            transform_inverse: Default::default(),
                            bounds: Default::default(),
                        },
                        parent: u32::MAX,
                        child_a: u32::MAX,
                        child_b: u32::MAX,
                    },
                );
            }
            cl.copy_buffer_to_device(
                staging_buffer,
                0,
                global_buffer,
                self.scene_lbvh_nodes_range.start() as u64,
                mem::size_of::<TopLBVHNode>() as u64,
            );
        } else {
            cl.debug_full_memory_barrier();
            cl.copy_buffer_to_device(
                staging_buffer,
                0,
                global_buffer,
                instances_range.start() as u64,
                instances_bytes.len() as u64,
            );

            cl.barrier_buffer(
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(instances_range.start() as u64)
                    .size(instances_range.len() as u64)
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)],
            );

            // 1. Calculate transformed instances bounds
            let payload = TLLBPayload {
                instances_offset: instances_range.start(),
                n_elements: n_instances as u32,
            };
            self.tl_leaf_bounds.dispatch(&mut cl, &payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(instances_range.start() as u64)
                    .size(instances_range.len() as u64)
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)],
            );

            // 2. Calculate scene bounding box
            let payload = TLBPayload {
                instances_offset: instances_range.start(),
                temp_aabbs_offset: temp_bounds_range.start(),
                n_elements: n_instances,
            };
            self.tl_bounds.dispatch(&mut cl, &payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(temp_bounds_range.start() as u64)
                    .size(temp_bounds_range.len() as u64)
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)],
            );

            // 3. Prepare top-level leaves
            let payload = TLPLPayload {
                morton_codes_offset: morton_codes_range.start(),
                instances_offset: instances_range.start(),
                top_nodes_offset: self.scene_lbvh_nodes_range.start(),
                scene_bounds_offset: temp_bounds_range.start(),
                n_leaves: n_instances,
            };
            self.tl_prepare_leaves.dispatch(&mut cl, &payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[
                    gb_barrier
                        .offset(morton_codes_range.start() as u64)
                        .size(morton_codes_len)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE),
                    gb_barrier
                        .offset(self.scene_lbvh_nodes_range.start() as u64)
                        .size(self.scene_lbvh_nodes_range.len() as u64)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE),
                ],
            );

            // 4. Sort morton codes
            let payload = MBSPayload {
                morton_codes_offset: morton_codes_range.start(),
                n_codes: n_instances,
            };
            self.bitonic_sort.dispatch(&mut cl, &payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(morton_codes_range.start() as u64)
                    .size(morton_codes_len)
                    .src_access_mask(AccessFlags::SHADER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ)],
            );

            // 5. Generate hierarchy
            let payload = TLGPayload {
                morton_codes_offset: morton_codes_range.start(),
                top_nodes_offset: self.scene_lbvh_nodes_range.start(),
                instances_offset: instances_range.start(),
                n_leaves: n_instances,
            };
            self.tl_gen.dispatch(&mut cl, &payload);

            cl.barrier_buffer(
                PipelineStageFlags::COMPUTE,
                PipelineStageFlags::COMPUTE | PipelineStageFlags::TRANSFER,
                &[
                    gb_barrier
                        .offset(self.scene_lbvh_nodes_range.start() as u64)
                        .size(self.scene_lbvh_nodes_range.len() as u64)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE),
                    gb_barrier
                        .offset(atomics_range.start() as u64)
                        .size(atomics_len)
                        .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::TRANSFER_WRITE),
                ],
            );

            // Fill atomics buffer with zeros for the step 6.
            cl.fill_buffer2(global_buffer, atomics_range.start() as u64, atomics_len, 0);

            cl.barrier_buffer(
                PipelineStageFlags::TRANSFER,
                PipelineStageFlags::COMPUTE,
                &[gb_barrier
                    .offset(atomics_range.start() as u64)
                    .size(atomics_len)
                    .src_access_mask(AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)],
            );

            // 6. Compute bounds for each internal node
            let payload = TLNBPayload {
                top_nodes_offset: self.scene_lbvh_nodes_range.start(),
                atomic_counters_offset: atomics_range.start(),
                n_leaves: n_instances,
            };
            cl.debug_full_memory_barrier();
            self.tl_node_bounds.dispatch(&mut cl, &payload);
        }

        // if !dirty_meshes.is_empty() {
        //     let gvb_mesh = &gvb_meshes[&dirty_meshes[0]];
        //     let n_triangles = gvb_mesh.raw.index_count / 3;
        //     let n_nodes = n_triangles * 2 - 1;
        //
        //     cl.debug_full_memory_barrier();
        //
        //     cl.copy_buffer_to_host(
        //         global_buffer,
        //         gvb_mesh.gb_rt_nodes_offset as u64,
        //         staging_buffer,
        //         0,
        //         mem::size_of::<LBVHNode>() as u64 * n_nodes as u64,
        //     );
        // }

        cl.end().unwrap();
        drop(cl);
        unsafe { graphics_queue.submit(staging_submit).unwrap() };
        staging_submit.wait().unwrap();

        // if !dirty_meshes.is_empty() {
        //     let gvb_mesh = &gvb_meshes[&dirty_meshes[0]];
        //     let n_triangles = gvb_mesh.raw.index_count / 3;
        //     let n_nodes = n_triangles * 2 - 1;
        //
        //     let nodes = unsafe {
        //         slice::from_raw_parts(staging_buffer.as_ptr() as *const LBVHNode, n_nodes as usize)
        //     };
        //
        //     for n in nodes {
        //         println!("{:?}", n);
        //     }
        // }

        // let top_nodes = unsafe {
        //     slice::from_raw_parts(
        //         staging_buffer.as_ptr().add(instances_bytes.len()) as *const TopLBVHNode,
        //         instances.len() * 2 - 1,
        //     )
        // };

        // let mut stack = vec![];
        // let mut curr_node_id = 0;
        //
        // while let Some(n) = stack.pop() {
        //     let node_a = top_nodes[n.child_a as usize];
        //     let node_b = top_nodes[n.child_b as usize];
        //     let traverse_a = node_a.instance.nodes_offset == u32::MAX;
        //     let traverse_b = node_b.instance.nodes_offset == u32::MAX;
        //
        //     if (!traverse_a && !traverse_b) {
        //     } else {
        //     }
        //
        //     if n.child_a != u32::MAX {
        //         stack.push();
        //     }
        //     if n.child_b != u32::MAX {
        //         stack.push(top_nodes[n.child_b as usize]);
        //     }
        // }

        // println!();
        // for n in &top_nodes[0..top_nodes.len().min(10)] {
        //     println!("{:?}", n);
        // }

        for alloc in allocs {
            global_buffer.free(alloc);
        }

        Ok(())
    }

    pub fn update_descriptors(&mut self, output_image: Arc<Image>, frame_info_buf: &DeviceBuffer) {
        self.ray_tracer.update_descriptors(output_image, frame_info_buf);
    }

    pub fn trace_rays(&self, cl: &mut CmdList) {
        let payload = RTPayload {
            top_nodes_offset: self.scene_lbvh_nodes_range.start(),
        };
        self.ray_tracer.dispatch(cl, &payload);
    }
}
