use crate::renderer::calc_group_count;
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{BindingRes, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer, Pipeline};

pub struct MortonCodesForTrianglesModule {
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
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/rt_morton_codes_for_triangles.comp.hlsl.spv"),
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
