use crate::renderer::helpers::{LargeBuffer, RendererError};
use crate::renderer::module::bottom_lbvh_generation::{BLGPayload, BottomLBVHGenModule};
use crate::renderer::module::bottom_lbvh_node_bounds::{BLNBPayload, BottomLBVHNodeBoundsModule};
use crate::renderer::module::bottom_lbvh_prepare_leaves::{BLPLPayload, BottomLBVHPrepareLeavesModule};
use crate::renderer::module::morton_bitonic_sort::{MBSPayload, MortonBitonicSortModule};
use crate::renderer::module::top_lbvh_bounds::TopLBVHBoundsModule;
use crate::renderer::module::top_lbvh_generation::TopLBVHGenModule;
use crate::renderer::module::top_lbvh_leaf_bounds::TopLBVHLeafBoundsModule;
use crate::renderer::module::top_lbvh_node_bounds::TopLBVHNodeBoundsModule;
use crate::renderer::module::top_lbvh_prepare_leaves::TopLBVHPrepareLeavesModule;
use crate::renderer::GBVertexMesh;
use crate::utils::HashMap;
use nalgebra_glm::{Mat4, Vec3};
use parking_lot::Mutex;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper::{AccessFlags, CmdList, Device, HostBuffer, PipelineStageFlags, Queue, SubmitPacket};

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

const BLBVH_MAX_TRIANGLES: u32 = 1024 * 1024;

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

#[derive(Debug)]
#[repr(C)]
struct MortonCode {
    code: u32,
    elem: u32,
}

#[derive(Debug, Copy, Clone)]
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
    pub fn new(device: &Arc<Device>, global_buffer: &LargeBuffer) -> Self {
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
        global_buffer: &mut LargeBuffer,
        staging_buffer: &mut HostBuffer<u8>,
        gvb_meshes: &HashMap<usize, GBVertexMesh>,
        dirty_meshes: &[usize],
    ) -> Result<(), RendererError> {
        if dirty_meshes.is_empty() {
            return Ok(());
        }
        let graphics_queue = self.device.get_queue(Queue::TYPE_GRAPHICS);
        let gb_barrier = global_buffer.barrier();

        let mut allocs = global_buffer.allocate_multiple(&[
            BLBVH_MAX_TRIANGLES * mem::size_of::<MortonCode>() as u32,
            BLBVH_MAX_TRIANGLES * mem::size_of::<u32>() as u32,
        ])?;
        let morton_codes_range = &allocs[0];
        let atomics_range = &allocs[1];

        // TODO: implement construction for non-indexed meshes

        let mut cl = staging_cl.lock();
        cl.begin(true).unwrap();

        let mut n_nodes = 0;
        let mut n_triangles = 0;
        let mut mesh_bounds_min = Vec3::default();
        let mut mesh_bounds_max = Vec3::default();

        for mesh_ptr in dirty_meshes {
            let gvb_mesh = &gvb_meshes[mesh_ptr];
            n_triangles = gvb_mesh.raw.index_count / 3;
            n_nodes = n_triangles * 2 - 1;

            let nodes_range_len = n_nodes as u64 * mem::size_of::<LBVHNode>() as u64;
            let morton_codes_len = n_triangles as u64 * mem::size_of::<MortonCode>() as u64;
            let atomics_len = n_triangles as u64 * mem::size_of::<u32>() as u64;

            if n_triangles > BLBVH_MAX_TRIANGLES {
                eprintln!(
                    "Max triangles for BLBVH is reached! Not processing this mesh. n_triangles = {}",
                    n_triangles
                );
                continue;
            }

            mesh_bounds_min = gvb_mesh.raw.aabb.0;
            mesh_bounds_max = gvb_mesh.raw.aabb.1;

            // Prepare leaves
            let blpl_payload = BLPLPayload {
                indices_offset: gvb_mesh.gb_indices_offset,
                vertices_offset: gvb_mesh.gb_binding_offsets[0],
                morton_codes_offset: morton_codes_range.start(),
                nodes_offset: gvb_mesh.gb_rt_nodes_offset,
                n_triangles,
                mesh_bound_min: gvb_mesh.raw.aabb.0,
                mesh_bound_max: gvb_mesh.raw.aabb.1,
            };
            self.bl_prepare_leaves.dispatch(&mut cl, &blpl_payload);

            // println!("{:?}", &gvb_mesh.raw.aabb);

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

            // Sort morton codes
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
            {
                cl.barrier_buffer(
                    PipelineStageFlags::COMPUTE,
                    PipelineStageFlags::TRANSFER,
                    &[gb_barrier
                        .offset(morton_codes_range.start() as u64)
                        .size(morton_codes_len)
                        .src_access_mask(AccessFlags::SHADER_WRITE)
                        .dst_access_mask(AccessFlags::TRANSFER_READ)],
                );

                cl.copy_buffer_to_host(
                    global_buffer,
                    morton_codes_range.start() as u64,
                    staging_buffer,
                    nodes_range_len,
                    morton_codes_len,
                );
            }

            // Generate LBVH
            let blg_payload = BLGPayload {
                morton_codes_offset: morton_codes_range.start(),
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

            // Compute node bounds
            let blnb_payload = BLNBPayload {
                nodes_offset: gvb_mesh.gb_rt_nodes_offset,
                atomic_counters_offset: atomics_range.start(),
                n_triangles,
            };
            // cl.debug_full_memory_barrier();
            self.bl_node_bounds.dispatch(&mut cl, &blnb_payload);

            {
                cl.barrier_buffer(
                    PipelineStageFlags::COMPUTE,
                    PipelineStageFlags::TRANSFER,
                    &[
                        gb_barrier
                            .offset(gvb_mesh.gb_rt_nodes_offset as u64)
                            .size(nodes_range_len)
                            .src_access_mask(AccessFlags::SHADER_WRITE)
                            .dst_access_mask(AccessFlags::TRANSFER_READ),
                        gb_barrier
                            .offset(atomics_range.start() as u64)
                            .size(atomics_len)
                            .src_access_mask(AccessFlags::SHADER_WRITE)
                            .dst_access_mask(AccessFlags::TRANSFER_READ),
                    ],
                );
                cl.copy_buffer_to_host(
                    global_buffer,
                    gvb_mesh.gb_rt_nodes_offset as u64,
                    staging_buffer,
                    0,
                    nodes_range_len,
                );
                cl.copy_buffer_to_host(
                    global_buffer,
                    atomics_range.start() as u64,
                    staging_buffer,
                    nodes_range_len + morton_codes_len,
                    atomics_len,
                );
            }

            break;
        }

        cl.end().unwrap();
        drop(cl);
        unsafe { graphics_queue.submit(staging_submit).unwrap() };
        staging_submit.wait().unwrap();

        let codes = unsafe {
            slice::from_raw_parts(
                staging_buffer
                    .as_ptr()
                    .add(n_nodes as usize * mem::size_of::<LBVHNode>()) as *const MortonCode,
                n_triangles as usize,
            )
        };

        for i in 0..codes.len().saturating_sub(1) {
            if codes[i].code > codes[i + 1].code {
                println!("{:?}", codes);
                panic!(
                    "incorrect sorting: left i{} = {}, right i{} = {}",
                    i,
                    codes[i].code,
                    i + 1,
                    codes[i + 1].code
                );
            }
        }

        // let atomics = unsafe {
        //     slice::from_raw_parts(
        //         staging_buffer.as_ptr().add(
        //             n_nodes as usize * mem::size_of::<LBVHNode>()
        //                 + n_triangles as usize * mem::size_of::<MortonCode>(),
        //         ) as *const u32,
        //         n_nodes as usize,
        //     )
        // };

        // assert_eq!(
        //     atomics.iter().filter(|v| **v == 2).count() as u32,
        //     n_triangles - 1
        // );

        let data =
            unsafe { slice::from_raw_parts(staging_buffer.as_ptr() as *const LBVHNode, n_nodes as usize) };
        let root = &data[0];
        let leaf_a = &data[root.child_a as usize];
        let leaf_b = &data[root.child_b as usize];
        let mut test_bound_min = Vec3::from_element(f32::MAX);
        let mut test_bound_max = Vec3::from_element(f32::MIN);
        let mut n_null_parents = 0;

        for i in 0..n_triangles {
            let mut leaf_id = n_triangles - 1 + i;

            let leaf = &data[leaf_id as usize];
            test_bound_min = test_bound_min.inf(&leaf.bounds.p_min);
            test_bound_max = test_bound_max.sup(&leaf.bounds.p_max);

            let par = &data[leaf.parent as usize];
            assert!(par.child_a == leaf_id || par.child_b == leaf_id);

            for j in 0..256 {
                let leaf = &data[leaf_id as usize];

                if leaf.parent == u32::MAX {
                    n_null_parents += 1;
                    break;
                } else if j == 255 {
                    println!("{:?}", &codes);
                    println!("{:?}", &data[..data.len().min(8)]);
                    // println!("{:?} {:?} {:?}", &root, &leaf_a, &leaf_b);

                    let f: Vec<_> = data
                        .iter()
                        .enumerate()
                        .filter(|(i, v)| v.child_a == 0 && v.child_b == 1)
                        .collect();
                    println!("{f:?}");

                    panic!("LOOP FOUND! leaf{}", i);
                }

                leaf_id = leaf.parent;
            }
        }

        assert_eq!(n_null_parents, n_triangles);

        let mut node_counts = vec![0; n_nodes as usize];

        for (i, node) in data.iter().enumerate() {
            if node.parent != u32::MAX {
                node_counts[node.parent as usize] += 1;
                assert!(node.parent < (n_triangles - 1))
            }

            if node.child_a != u32::MAX {
                assert_eq!(data[node.child_a as usize].parent, i as u32);
            }
            if node.child_b != u32::MAX {
                assert_eq!(data[node.child_b as usize].parent, i as u32);
            }
        }

        // println!("{:?}", &node_counts);

        for c in node_counts {
            assert!(c == 2 || c == 0);
        }

        if root.bounds.p_min != mesh_bounds_min || root.bounds.p_max != mesh_bounds_max {
            // println!("{:?}", &data[..data.len().min(8)]);
            // println!("{:?}", &data[..data.len()]);
            // println!("{:?}", &data[data.len().saturating_sub(8)..]);
            // println!("{:?}", atomics);
            println!(
                "{:?} {:?} | {:?} {:?}",
                mesh_bounds_min, mesh_bounds_max, root.bounds.p_min, root.bounds.p_max
            );

            for (i, d) in data.iter().enumerate() {
                println!("{} - {:?}", i, d);
            }

            // let mut parent_bounds = HashMap::with_capacity(n_triangles as usize);

            // for i in 0..n_triangles {
            //     let mut leaf_id = n_triangles - 1 + i;
            //     // let leaf = &data[leaf_id as usize];
            //     // let parent = &data[leaf.parent as usize];
            //     // let parent2 = &data[parent.parent as usize];
            //
            //     for i in 0..4 {
            //         let leaf = &data[leaf_id as usize];
            //         leaf_id = leaf.parent;
            //         if leaf_id == u32::MAX {
            //             break;
            //         }
            //     }
            //
            //     let leaf = &data[leaf_id as usize];
            //
            //     // assert!(parent.parent < n_triangles - 1);
            //     parent_bounds.insert(leaf_id, leaf.bounds);
            // }

            // println!(
            //     "PARENT BOUNDS: {:?} parents{} n_triangles{}",
            //     parent_bounds.values().collect::<Vec<_>>(),
            //     parent_bounds.len(),
            //     n_triangles
            // );

            let last = data.last().unwrap();
            let parent = &data[last.parent as usize];
            println!("LAST PARENT BOUNDS: {:?}", parent.bounds);
            println!("TEST BOUNDS: {:?} {:?}", test_bound_min, test_bound_max);

            assert_eq!(root.bounds.p_min, mesh_bounds_min);
            assert_eq!(root.bounds.p_max, mesh_bounds_max);
        }

        // println!("{:?} {:?}", mesh_bounds_min, mesh_bounds_max);
        // println!("{:?}", &data[..data.len().min(8)]);

        // assert_eq!(root.bounds.p_min, mesh_bounds_min);
        // assert_eq!(root.bounds.p_max, mesh_bounds_max);

        //
        // if leaf_a.parent != 0 || leaf_b.parent != 0 {
        //     println!("{:?}", &codes);
        //     println!("{:?}", &data);
        //     println!();
        //     println!("{:?}", &data[0]);
        //     println!("{:?}", &data[data[0].child_a as usize]);
        //     assert_eq!(leaf_a.parent, 0);
        // }

        // println!("{:?} {:?} {:?}", &root, &leaf_a, &leaf_b);

        for alloc in allocs {
            global_buffer.free(alloc);
        }

        Ok(())
    }
}
