use crate::utils::UInt;
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    Pipeline, PipelineStageFlags,
};

const WORK_GROUP_SIZE: u32 = 1024;

pub struct MortonBitonicSortModule {
    pipeline: Arc<Pipeline>,
    pool: DescriptorPool,
    descriptor: DescriptorSet,
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
struct PushConstants {
    h: u32,
    algorithm: MBSAlgorithm,
    n_values: u32,
}

impl MortonBitonicSortModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_morton_bitonic_sort.comp.hlsl.spv"),
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
                &[pool.create_binding(0, 0, BindingRes::Buffer(global_buffer.handle()))],
            );
        }

        Self {
            pipeline,
            pool,
            descriptor,
            gb_barrier: global_buffer.barrier(),
        }
    }

    fn dispatch(&self, cl: &mut CmdList, payloads: &[MBSPayload]) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        for payload in payloads {
            let mut h = WORK_GROUP_SIZE * 2;
            let work_group_count = UInt::div_ceil(payload.n_codes, WORK_GROUP_SIZE * 2);

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
                let mut consts = PushConstants {
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
                    if hh <= WORK_GROUP_SIZE * 2 {
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
