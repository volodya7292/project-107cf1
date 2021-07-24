use crate::render_engine::material_pipeline::MaterialPipeline;
use crate::render_engine::RenderEngine;
use crate::resource_file::ResourceFile;
use nalgebra as na;
use std::sync::Arc;
use vk_wrapper as vkw;

pub struct MaterialPipelines {
    triag: u32,
    cluster: u32,
}

impl MaterialPipelines {
    pub fn triag(&self) -> u32 {
        self.triag
    }

    pub fn cluster(&self) -> u32 {
        self.cluster
    }
}

pub fn create(resources: &Arc<ResourceFile>, renderer: &mut RenderEngine) -> MaterialPipelines {
    let device = Arc::clone(renderer.device());

    let triag = {
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

        renderer.register_material_pipeline::<BasicUniformInfo>(&[triag_vertex, triag_g_pixel])
    };
    let cluster = {
        let vertex = device
            .create_shader(
                &resources.get("shaders/cluster.vert.spv").unwrap().read().unwrap(),
                &[
                    ("inPosition", vkw::Format::RGB32_FLOAT),
                    ("inNormal", vkw::Format::RGB32_FLOAT),
                    ("inTexUV", vkw::Format::RG32_FLOAT),
                    ("inAO", vkw::Format::R32_FLOAT),
                    ("inMaterialId", vkw::Format::R32_UINT),
                ],
                &[],
            )
            .unwrap();
        let pixel = device
            .create_shader(
                &resources.get("shaders/cluster.frag.spv").unwrap().read().unwrap(),
                &[],
                &[],
            )
            .unwrap();

        renderer.register_material_pipeline::<BasicUniformInfo>(&[vertex, pixel])
    };

    MaterialPipelines { triag, cluster }
}

#[derive(Default)]
pub struct BasicUniformInfo {
    model: na::Matrix4<f32>,
}

uniform_struct_impl!(BasicUniformInfo, model);
