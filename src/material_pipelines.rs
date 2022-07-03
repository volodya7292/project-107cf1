use engine::renderer::Renderer;
use engine::resource_file::ResourceFile;
use engine::uniform_struct_impl;
use nalgebra as na;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::shader::VInputRate;

pub struct MaterialPipelines {
    cluster: u32,
}

impl MaterialPipelines {
    pub fn cluster(&self) -> u32 {
        self.cluster
    }
}

fn create_vertex_shader(
    device: &Arc<vkw::Device>,
    code: &[u8],
    input_formats: &[(&str, vkw::Format)],
) -> Result<Arc<vkw::Shader>, vkw::DeviceError> {
    device.create_vertex_shader(
        code,
        &input_formats
            .iter()
            .cloned()
            .map(|(name, format)| (name, format, VInputRate::VERTEX))
            .collect::<Vec<_>>(),
        &[("per_object_data", vkw::ShaderBindingMod::DYNAMIC_OFFSET)],
    )
}

pub fn create(resources: &Arc<ResourceFile>, renderer: &mut Renderer) -> MaterialPipelines {
    let device = Arc::clone(renderer.device());

    let cluster = {
        let vertex = create_vertex_shader(
            &device,
            &resources.get("shaders/cluster.vert.spv").unwrap().read().unwrap(),
            &[
                ("inPack1", vkw::Format::RGBA32_UINT),
                ("inPack2", vkw::Format::R32_UINT),
            ],
        )
        .unwrap();
        let pixel = device
            .create_pixel_shader(
                &resources.get("shaders/cluster.frag.spv").unwrap().read().unwrap(),
                &[],
            )
            .unwrap();

        renderer.register_material_pipeline::<BasicUniformInfo>(&[vertex, pixel])
    };

    MaterialPipelines { cluster }
}

#[derive(Default)]
pub struct BasicUniformInfo {
    model: na::Matrix4<f32>,
}

uniform_struct_impl!(BasicUniformInfo, model);
