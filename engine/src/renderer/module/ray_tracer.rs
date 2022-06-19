use crate::renderer::calc_group_count_2d;
use nalgebra_glm as glm;
use nalgebra_glm::{UVec2, Vec2};
use std::sync::Arc;
use vk_wrapper::buffer::BufferHandleImpl;
use vk_wrapper::{
    BindingRes, CmdList, DescriptorPool, DescriptorSet, Device, DeviceBuffer, Image, ImageLayout, Pipeline,
};

pub struct RayTracerModule {
    device: Arc<Device>,
    pipeline: Arc<Pipeline>,
    pool: DescriptorPool,
    descriptor: DescriptorSet,
    resolution: Vec2,
}

#[repr(C)]
pub struct RTPayload {
    pub top_nodes_offset: u32,
}

#[repr(C)]
struct PushConstants {
    pub resolution: Vec2,
    pub top_nodes_offset: u32,
}

impl RayTracerModule {
    pub fn new(device: &Arc<Device>, global_buffer: &DeviceBuffer) -> Self {
        let shader = device
            .create_shader(
                include_bytes!("../../../shaders/build/trace_rays.comp.hlsl.spv"),
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
            device: Arc::clone(device),
            pipeline,
            pool,
            descriptor,
            resolution: Default::default(),
        }
    }

    pub fn update_descriptors(&mut self, output_image: Arc<Image>, frame_info_buf: &DeviceBuffer) {
        unsafe {
            self.device.update_descriptor_set(
                self.descriptor,
                &[
                    self.pool.create_binding(
                        1,
                        0,
                        BindingRes::ImageView(Arc::clone(output_image.view()), ImageLayout::GENERAL),
                    ),
                    self.pool
                        .create_binding(2, 0, BindingRes::Buffer(frame_info_buf.handle())),
                ],
            );
        }
        let size = output_image.size_2d();
        self.resolution = Vec2::new(size.0 as f32, size.1 as f32);
    }

    pub fn dispatch(&self, cl: &mut CmdList, payload: &RTPayload) {
        cl.bind_pipeline(&self.pipeline);
        cl.bind_compute_input(self.pipeline.signature(), 0, self.descriptor, &[]);

        let consts = PushConstants {
            resolution: self.resolution,
            top_nodes_offset: payload.top_nodes_offset,
        };
        let ires: UVec2 = glm::try_convert(self.resolution).unwrap();
        let groups_x = calc_group_count_2d(ires.x);
        let groups_y = calc_group_count_2d(ires.y);

        cl.push_constants(self.pipeline.signature(), &consts);
        cl.dispatch(groups_x, groups_y, 1);
    }
}
