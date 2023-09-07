use common::resource_file::BufferedResourceReader;
use engine::module::main_renderer::shader::{VkwShaderBundle, VkwShaderBundleDeviceExt};
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::text_renderer::{self, TextRenderer};
use engine::vkw::pipeline::CullMode;
use engine::vkw::shader::VInputRate;
use engine::vkw::PrimitiveTopology;
use engine::{vkw, EngineContext};
use std::sync::Arc;

pub struct MaterialPipelines {
    pub cluster: MaterialPipelineId,
    pub text_3d: MaterialPipelineId,
}

fn load_vertex_shader_bundle(
    device: &Arc<vkw::Device>,
    bundle_data: &[u8],
    input_formats: &[(&str, vkw::Format)],
    name: &str,
) -> Result<Arc<VkwShaderBundle>, vkw::DeviceError> {
    device.load_vertex_shader_bundle(
        bundle_data,
        &input_formats
            .iter()
            .cloned()
            .map(|(name, format)| (name, format, VInputRate::VERTEX))
            .collect::<Vec<_>>(),
        name,
    )
}

pub fn create(resources: &Arc<BufferedResourceReader>, ctx: &EngineContext) -> MaterialPipelines {
    let mut renderer = ctx.module_mut::<MainRenderer>();
    let mut text_renderer = ctx.module_mut::<TextRenderer>();

    let device = Arc::clone(renderer.device());

    let cluster = {
        let vertex = load_vertex_shader_bundle(
            &device,
            &resources.get("shaders/cluster.vert.b").unwrap(),
            &[
                ("inPack1", vkw::Format::RGBA32_UINT),
                ("inPack2", vkw::Format::R32_UINT),
            ],
            "cluster.vert",
        )
        .unwrap();
        let pixel = device
            .load_pixel_shader_bundle(&resources.get("shaders/cluster.frag.b").unwrap(), "cluster.frag")
            .unwrap();

        renderer.register_material_pipeline(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_LIST,
            CullMode::BACK,
        )
    };

    let text_3d = {
        let vertex = device
            .load_vertex_shader_bundle(
                &resources.get("shaders/text_char.vert.b").unwrap(),
                &text_renderer::VERTEX_INPUTS,
                "text_char.vert",
            )
            .unwrap();
        let pixel = device
            .load_pixel_shader_bundle(
                &resources.get("shaders/text_char_3d.frag.b").unwrap(),
                "text_char_3d.frag",
            )
            .unwrap();

        text_renderer.register_text_pipeline(&mut renderer, &[vertex, pixel])
    };

    MaterialPipelines { cluster, text_3d }
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
pub struct BasicUniformInfo {}
