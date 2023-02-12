use base::utils::resource_file::ResourceFile;
use engine::module::main_renderer::MainRenderer;
use engine::uniform_struct_impl;
use nalgebra as na;
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::shader::VInputRate;
use vk_wrapper::PrimitiveTopology;

pub struct MaterialPipelines {
    cluster: u32,
    panel: u32,
}

impl MaterialPipelines {
    pub fn cluster(&self) -> u32 {
        self.cluster
    }

    pub fn panel(&self) -> u32 {
        self.panel
    }
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

pub fn create(resources: &Arc<ResourceFile>, renderer: &mut MainRenderer) -> MaterialPipelines {
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

        renderer.register_material_pipeline::<BasicUniformInfo>(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_LIST,
            true,
        )
    };

    let panel = {
        let vertex = create_vertex_shader(
            &device,
            // &resources
            //     .get("../../engine/shaders/build/ui_rect.vert.spv")
            //     .unwrap()
            //     .read()
            //     .unwrap(),
            include_bytes!("../../engine/shaders/build/ui_rect.vert.spv"),
            &[],
            "ui_rect.vert",
        )
        .unwrap();
        let pixel = device
            .create_pixel_shader(
                include_bytes!("../../engine/shaders/build/panel.frag.spv"),
                // &resources
                //     .get("../../engine/shaders/build/panel.frag.spv")
                //     .unwrap()
                //     .read()
                //     .unwrap(),
                "panel.frag",
            )
            .unwrap();

        renderer.register_material_pipeline::<BasicUniformInfo>(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_STRIP,
            true,
        )
    };

    MaterialPipelines { cluster, panel }
}

#[derive(Default)]
#[repr(C)]
pub struct BasicUniformInfo {
    model: na::Matrix4<f32>,
}

uniform_struct_impl!(BasicUniformInfo, model);
