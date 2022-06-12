use crate::renderer::calc_group_count_1d;
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{BindingRes, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer, Pipeline};

pub struct TopLBVHLeafBoundsModule {
    pipeline: Arc<Pipeline>,
    pool: DescriptorPool,
    descriptor: DescriptorSet,
}

#[repr(C)]
pub struct TLLBPayload {
    pub instances_offset: u32,
    pub n_elements: u32,
}

impl TopLBVHLeafBoundsModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_top_lbvh_leaf_bounds.comp.hlsl.spv"),
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
        }
    }

    pub fn dispatch(&self, cl: &mut CmdList, payload: &TLLBPayload) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        let groups = calc_group_count_1d(payload.n_elements);
        cl.push_constants(self.pipeline.signature(), payload);
        cl.dispatch(groups, 1, 1);
    }
}