use crate::renderer::material_pipeline::MaterialPipeline;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::{Device, DeviceBuffer, PipelineInput};

pub struct Renderer {
    pub(in crate::renderer) mat_pipeline: Arc<MaterialPipeline>,
    pub(in crate::renderer) pipeline_inputs: Vec<Arc<PipelineInput>>,

    pub(in crate::renderer) uniform_buffer: Arc<DeviceBuffer>,
    //buffers: HashMap<u32, vkw::RawHostBuffer>,
    // binding id -> renderer impl-specific res index
    pub(in crate::renderer) translucent: bool,
    pub(in crate::renderer) changed: bool,
}

impl Renderer {
    pub fn new(device: &Arc<Device>, mat_pipeline: &Arc<MaterialPipeline>, translucent: bool) -> Renderer {
        Self {
            mat_pipeline: Arc::clone(&mat_pipeline),
            pipeline_inputs: vec![],
            uniform_buffer: device
                .create_device_buffer(
                    vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::UNIFORM,
                    mat_pipeline.uniform_buffer_size() as u64,
                    1,
                )
                .unwrap(),
            translucent,
            changed: true,
        }
    }
}
