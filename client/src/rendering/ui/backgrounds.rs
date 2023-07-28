use crate::rendering::ui::container::ContainerBackground;
use common::make_static_id;
use engine::module::main_renderer::MainRenderer;
use engine::module::scene::Scene;
use engine::module::ui::color::Color;
use engine::vkw::pipeline::CullMode;
use engine::vkw::PrimitiveTopology;
use engine::EngineContext;

mod fancy {
    use super::*;
    use common::glm::Vec4;

    pub const MATERIAL_PIPE_RES_NAME: &str = make_static_id!();

    #[derive(Default, Copy, Clone)]
    #[repr(C)]
    pub struct UniformData {
        pub background_color: Vec4,
    }

    pub fn register(ctx: &EngineContext) {
        let mut renderer = ctx.module_mut::<MainRenderer>();
        let scene = ctx.module_mut::<Scene>();

        let vertex = renderer
            .device()
            .create_vertex_shader(
                include_bytes!("../../../res/shaders/ui_rect.vert.spv"),
                &[],
                "ui_rect.vert",
            )
            .unwrap();
        let pixel = renderer
            .device()
            .create_pixel_shader(
                include_bytes!("../../../res/shaders/fancy_background.frag.spv"),
                "fancy_background.frag",
            )
            .unwrap();

        let mat_pipe_id = renderer.register_material_pipeline(
            &[vertex, pixel],
            PrimitiveTopology::TRIANGLE_STRIP,
            CullMode::BACK,
        );

        scene.register_named_resource(MATERIAL_PIPE_RES_NAME, mat_pipe_id);
    }
}

pub fn register(ctx: &EngineContext) {
    fancy::register(ctx);
}

pub const DEFAULT_FANCY_COLOR: Color = Color::rgb(0.2, 0.3, 0.2);

pub fn fancy(color: Color) -> ContainerBackground {
    ContainerBackground::new_raw(
        fancy::MATERIAL_PIPE_RES_NAME,
        fancy::UniformData {
            background_color: color.into_raw(),
        },
    )
}
