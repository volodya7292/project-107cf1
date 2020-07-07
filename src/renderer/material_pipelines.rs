use crate::renderer::material_pipeline;
use crate::renderer::material_pipeline::MaterialPipeline;
use nalgebra as na;
use std::sync::Arc;
use vk_wrapper as vkw;

pub struct MaterialPipelines {
    basic: Arc<MaterialPipeline>,
}

impl MaterialPipelines {
    pub fn basic(&self) -> Arc<MaterialPipeline> {
        Arc::clone(&self.basic)
    }
}

pub fn create(device: &Arc<vkw::Device>) -> MaterialPipelines {
    let basic = material_pipeline::new::<BasicUniformInfo>();

    MaterialPipelines { basic }
}

#[derive(Default)]
pub struct BasicUniformInfo {
    some_shit: na::Vector4<f32>,
    model: na::Matrix4<f32>,
}

uniform_struct_impl!(BasicUniformInfo, model);
