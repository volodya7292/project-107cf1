use crate::renderer::{calc_group_count, calc_group_count2};
use crate::Renderer;
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    DeviceError, Pipeline, PipelineSignature, PipelineStageFlags,
};

struct MortonCodesForTrianglesModule {
    pipeline: Arc<Pipeline>,
    pool: DescriptorPool,
    descriptor: DescriptorSet,
}

#[repr(C)]
struct MCTPayload {
    indices_offset: u32,
    vertices_offset: u32,
    morton_codes_offset: u32,
    n_triangles: u32,
}

impl MortonCodesForTrianglesModule {
    fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_morton_codes_for_triangles.comp.spv"),
                &[],
                &[],
            )
            .unwrap();
        let signature = device.create_pipeline_signature(&[shader], &[]).unwrap();
        let pipeline = device.create_compute_pipeline(&signature, &[]).unwrap();

        let mut pool = signature.create_pool(0, 1).unwrap();
        let descriptor = pool.alloc().unwrap();
        unsafe {
            device.update_descriptor_set(
                descriptor,
                &[
                    pool.create_binding(0, 0, BindingRes::Buffer(global_buffer.handle())),
                    pool.create_binding(1, 0, BindingRes::Buffer(global_buffer.handle())),
                    pool.create_binding(2, 0, BindingRes::Buffer(global_buffer.handle())),
                ],
            );
        }

        Self {
            pipeline,
            pool,
            descriptor,
        }
    }

    fn dispatch(&self, cl: &mut CmdList, payloads: &[MCTPayload]) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        for payload in payloads {
            let groups = calc_group_count(payload.n_triangles);
            cl.push_constants(self.pipeline.signature(), payload);
            cl.dispatch(groups, 1, 1);
        }
    }
}

struct MortonBitonicSort {
    pipeline: Arc<Pipeline>,
    pool: DescriptorPool,
    descriptor: DescriptorSet,
    work_group_size_x: u32,
    gb_barrier: BufferBarrier,
}

#[repr(u32)]
enum MBSAlgorithm {
    LocalBitonicMergeSort = 0,
    LocalDisperse = 1,
    BigFlip = 2,
    BigDisperse = 3,
}

struct MBSPayload {
    morton_codes_offset: u32,
    n_codes: u32,
}

#[repr(C)]
struct MBSPushConstants {
    h: u32,
    algorithm: MBSAlgorithm,
    n_values: u32,
}

impl MortonBitonicSort {
    fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_morton_bitonic_sort.comp.spv"),
                &[],
                &[],
            )
            .unwrap();
        let signature = device.create_pipeline_signature(&[shader], &[]).unwrap();

        // Note: allow maximum of 1024 threads per work group.
        // Minimum shared memory is 16384 bytes => 16384 / 8 = 2048 uvec2 elements.
        // Therefore divide by 2 (see glsl shared memory size) to get a limit of 1024 threads.
        let max_work_group_size = device.adapter().props().limits.max_compute_work_group_size;
        let work_group_size_x = max_work_group_size[0].min(1024);

        let pipeline = device
            .create_compute_pipeline(&signature, &[(0, work_group_size_x)])
            .unwrap();

        let mut pool = signature.create_pool(0, 1).unwrap();
        let descriptor = pool.alloc().unwrap();
        unsafe {
            device.update_descriptor_set(
                descriptor,
                &[pool.create_binding(0, 0, BindingRes::Buffer(global_buffer.handle()))],
            );
        }

        Self {
            pipeline,
            pool,
            descriptor,
            work_group_size_x,
            gb_barrier: global_buffer.barrier(),
        }
    }

    fn dispatch(&self, cl: &mut CmdList, payloads: &[MBSPayload]) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        for payload in payloads {
            let mut h = self.work_group_size_x * 2;
            let work_group_count = calc_group_count2(payload.n_codes, self.work_group_size_x * 2);

            let buf_offset = payload.morton_codes_offset as u64;
            let buf_size = (payload.n_codes * 8) as u64; // sizeof(uvec2) = 8
            let buf_barrier = self
                .gb_barrier
                .clone()
                .offset(buf_offset)
                .size(buf_size)
                .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

            let mut execute = |h: u32, alg: MBSAlgorithm| {
                let mut consts = MBSPushConstants {
                    h,
                    algorithm: alg,
                    n_values: payload.n_codes,
                };
                cl.push_constants(self.pipeline.signature(), &consts);
                cl.dispatch(work_group_count, 1, 1);
                cl.barrier_buffer(
                    PipelineStageFlags::COMPUTE,
                    PipelineStageFlags::COMPUTE,
                    &[buf_barrier],
                );
            };

            execute(h, MBSAlgorithm::LocalBitonicMergeSort);
            h *= 2;

            while h <= payload.n_codes {
                execute(h, MBSAlgorithm::BigFlip);

                let mut hh = h / 2;
                while hh > 1 {
                    if hh <= self.work_group_size_x * 2 {
                        execute(hh, MBSAlgorithm::LocalDisperse);
                    } else {
                        execute(hh, MBSAlgorithm::BigDisperse);
                    }
                    hh /= 2;
                }

                h *= 2;
            }
        }
    }
}

pub struct RayTracingModule {
    mct: MortonCodesForTrianglesModule,
    mbs: MortonBitonicSort,
}

impl RayTracingModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        Self {
            mct: MortonCodesForTrianglesModule::new(device, global_buffer),
            mbs: MortonBitonicSort::new(device, global_buffer),
        }
    }

    fn compute_bvh(&self) {}
}
