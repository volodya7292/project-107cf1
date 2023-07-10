use common::resource_file::ResourceFile;
use engine::module::main_renderer::{MainRenderer, MaterialPipelineId};
use engine::module::text_renderer::TextRenderer;
use engine::vkw::pipeline::CullMode;
use engine::vkw::shader::VInputRate;
use engine::vkw::PrimitiveTopology;
use engine::{vkw, EngineContext};
use std::sync::Arc;

pub struct MaterialPipelines {
    pub cluster: MaterialPipelineId,
    pub text_3d: MaterialPipelineId,
}

fn create_vertex_shader(
    device: &Arc<vkw::Device>,
    code: &[u8],
    input_formats: &[(&str, vkw::Format)],
    name: &str,
) -> Result<Arc<vkw::Shader>, vkw::DeviceError> {
    device.create_vertex_shader(
        code,
        &input_formats
            .iter()
            .cloned()
            .map(|(name, format)| (name, format, VInputRate::VERTEX))
            .collect::<Vec<_>>(),
        name,
    )
}

pub fn create(resources: &Arc<ResourceFile>, ctx: &EngineContext) -> MaterialPipelines {
    let mut renderer = ctx.module_mut::<MainRenderer>();
    let mut text_renderer = ctx.module_mut::<TextRenderer>();

    let device = Arc::clone(renderer.device());

    let cluster = {
        let vertex = create_vertex_shader(
            &device,
            &resources.get("shaders/cluster.vert.spv").unwrap().read().unwrap(),
            &[
                ("inPack1", vkw::Format::RGBA32_UINT),
                ("inPack2", vkw::Format::R32_UINT),
            ],
            "cluster.vert",
        )
        .unwrap();
        let pixel = device
            .create_pixel_shader(
                &resources.get("shaders/cluster.frag.spv").unwrap().read().unwrap(),
                "cluster.frag",
            )
            .unwrap();

        renderer.register_material_pipeline(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_LIST,
            CullMode::BACK,
        )
    };

    let text_3d = {
        let pixel = device
            .create_pixel_shader(
                include_bytes!("../../res/shaders/text_char_3d.frag.spv"),
                // &resources
                //     .get("shaders/text_char_3d.frag.spv")
                //     .unwrap()
                //     .read()
                //     .unwrap(),
                "text_char_3d.frag",
            )
            .unwrap();

        text_renderer.register_text_pipeline(&mut renderer, pixel)
    };

    MaterialPipelines { cluster, text_3d }
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
pub struct BasicUniformInfo {}
