use crate::renderer::module::ray_tracing::Bounds;
use crate::utils::UInt;
use std::mem;
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    AccessFlags, BindingRes, BufferBarrier, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer,
    Pipeline, PipelineStageFlags,
};

const WORK_GROUP_SIZE: u32 = 512;

pub struct TopLBVHBoundsModule {
    pipeline: Arc<Pipeline>,
    pool: DescriptorPool,
    descriptor: DescriptorSet,
    gb_barrier: BufferBarrier,
}

struct TLBPayload {
    top_nodes_offset: u32,
    temp_aabbs_offset: u32,
    n_elements: u32,
}

#[repr(C)]
struct PushConstants {
    top_nodes_offset: u32,
    temp_aabbs_offset: u32,
    n_elements: u32,
    iteration: u32,
}

impl TopLBVHBoundsModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_top_lbvh_bounds.comp.hlsl.spv"),
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

    fn dispatch(&self, cl: &mut CmdList, payloads: &[TLBPayload]) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        for payload in payloads {
            let iter_count = UInt::log(payload.n_elements, WORK_GROUP_SIZE);

            let buf_barrier = self
                .gb_barrier
                .clone()
                .offset(payload.temp_aabbs_offset as u64)
                .size(iter_count as u64 * mem::size_of::<Bounds>() as u64)
                .src_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE)
                .dst_access_mask(AccessFlags::SHADER_READ | AccessFlags::SHADER_WRITE);

            let mut n_elements = payload.n_elements;

            for i in 0..iter_count {
                let mut consts = PushConstants {
                    top_nodes_offset: payload.top_nodes_offset,
                    temp_aabbs_offset: payload.temp_aabbs_offset,
                    n_elements: payload.n_elements,
                    iteration: i,
                };

                let work_group_count = UInt::div_ceil(n_elements, WORK_GROUP_SIZE);

                cl.push_constants(self.pipeline.signature(), &consts);
                cl.dispatch(work_group_count, 1, 1);
                cl.barrier_buffer(
                    PipelineStageFlags::COMPUTE,
                    PipelineStageFlags::COMPUTE,
                    &[buf_barrier],
                );

                n_elements = work_group_count;
            }
        }
    }
}
