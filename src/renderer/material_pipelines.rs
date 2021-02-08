use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::Renderer;
use crate::resource_file::ResourceFile;
use nalgebra as na;
use std::sync::Arc;
use vk_wrapper as vkw;

pub struct MaterialPipelines {
    triag: Arc<MaterialPipeline>,
    cluster: Arc<MaterialPipeline>,
}

impl MaterialPipelines {
    pub fn triag(&self) -> Arc<MaterialPipeline> {
        Arc::clone(&self.triag)
    }

    pub fn cluster(&self) -> Arc<MaterialPipeline> {
        Arc::clone(&self.cluster)
    }
}

pub fn create(resources: &Arc<ResourceFile>, renderer: &mut Renderer) -> MaterialPipelines {
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

        renderer.create_material_pipeline::<BasicUniformInfo>(&[triag_vertex, triag_g_pixel])
    };
    let cluster = {
        let vertex = device
            .create_shader(
                &resources.get("shaders/cluster.vert.spv").unwrap().read().unwrap(),
                &[
                    ("inPosition", vkw::Format::RGB32_FLOAT),
                    ("inNormal", vkw::Format::RGB32_FLOAT),
                    ("inMaterialId", vkw::Format::R32_UINT),
                ],
                &[],
            )
            .unwrap();
        let geom = device
            .create_shader(
                &resources.get("shaders/cluster.geom.spv").unwrap().read().unwrap(),
                &[],
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

        renderer.create_material_pipeline::<BasicUniformInfo>(&[vertex, geom, pixel])
    };

    MaterialPipelines { triag, cluster }
}

#[derive(Default)]
pub struct BasicUniformInfo {
    model: na::Matrix4<f32>,
}

uniform_struct_impl!(BasicUniformInfo, model);
