use crate::renderer::material_pipeline;
use crate::renderer::material_pipeline::MaterialPipeline;
use crate::resource_file::ResourceFile;
use nalgebra as na;
use std::sync::Arc;
use vk_wrapper as vkw;

pub struct MaterialPipelines {
    triag: Arc<MaterialPipeline>,
}

impl MaterialPipelines {
    pub fn triag(&self) -> Arc<MaterialPipeline> {
        Arc::clone(&self.triag)
    }
}

pub fn create(resources: &Arc<ResourceFile>, device: &Arc<vkw::Device>) -> MaterialPipelines {
    let triag_vertex = device
        .create_shader(
            &resources.get("shaders/triag.vert.spv").unwrap().read().unwrap(),
            &[
                ("inPosition", vkw::Format::RGB32_FLOAT),
                ("inTexCoord", vkw::Format::RG32_FLOAT),
            ],
            &[],
        )
        .unwrap();
    let triag_g_pixel = device
        .create_shader(
            &resources.get("shaders/triag.frag.spv").unwrap().read().unwrap(),
            &[],
            &[],
        )
        .unwrap();
    let triag = material_pipeline::new::<BasicUniformInfo>(device, &triag_vertex, &triag_g_pixel);

    MaterialPipelines { triag }
}

#[derive(Default)]
pub struct BasicUniformInfo {
    model: na::Matrix4<f32>,
}

uniform_struct_impl!(BasicUniformInfo, model);
