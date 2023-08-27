use crate::rendering::ui::container::ContainerBackground;
use common::make_static_id;
use engine::ecs::component::render_config::GPUImageResource;
use engine::module::main_renderer::MainRenderer;
use engine::module::scene::Scene;
use engine::module::ui::color::Color;
use engine::vkw::pipeline::CullMode;
use engine::vkw::PrimitiveTopology;
use engine::EngineContext;

use super::image::ImageSource;

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

mod game_effects {
    use super::*;
    use common::glm::Vec4;
    use engine::vkw::utils::GLSLBool;

    pub const MATERIAL_PIPE_RES_NAME: &str = make_static_id!();

    #[derive(Default, Copy, Clone)]
    #[repr(C)]
    pub struct UniformData {
        pub filter: Vec4,
        // [0;1]
        pub pain_factor: f32,
        pub vision_obstructed: GLSLBool,
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
                include_bytes!("../../../res/shaders/game_effects_background.frag.spv"),
                "game_effects_background.frag",
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

mod health_indicators {
    use super::*;

    pub const MATERIAL_PIPE_RES_NAME: &str = make_static_id!();

    #[derive(Default, Copy, Clone)]
    #[repr(C)]
    pub struct UniformData {
        pub health_factor: f32,
        pub satiety_factor: f32,
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
                include_bytes!("../../../res/shaders/ui_background_health_indicators.frag.spv"),
                "ui_background_health_indicators.frag",
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

mod hud_popup {
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
                include_bytes!("../../../res/shaders/ui_background_hud_popup.frag.spv"),
                "ui_background_slot_circle.frag",
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

mod item_slot {
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
                include_bytes!("../../../res/shaders/ui_background_item_slot.frag.spv"),
                "ui_background_item_slot.frag",
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

mod material_item {
    use super::*;
    use common::glm::Vec4;

    pub const MATERIAL_PIPE_RES_NAME: &str = make_static_id!();

    #[derive(Default, Copy, Clone)]
    #[repr(C)]
    pub struct UniformData {
        // pub background_color: Vec4,
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
                include_bytes!("../../../res/shaders/ui_background_material_item.frag.spv"),
                "ui_background_material_item.frag",
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

pub fn fancy(color: Color) -> ContainerBackground {
    ContainerBackground::new_raw(
        fancy::MATERIAL_PIPE_RES_NAME,
        fancy::UniformData {
            background_color: color.into_raw_linear(),
        },
        vec![],
    )
}

pub fn game_effects(filter: Color, pain_factor: f32, vision_obstructed: bool) -> ContainerBackground {
    ContainerBackground::new_raw(
        game_effects::MATERIAL_PIPE_RES_NAME,
        game_effects::UniformData {
            filter: filter.into_raw_linear(),
            pain_factor,
            vision_obstructed: vision_obstructed as u32,
        },
        vec![],
    )
}

pub fn health_indicators(health_factor: f32, satiety_factor: f32) -> ContainerBackground {
    ContainerBackground::new_raw(
        health_indicators::MATERIAL_PIPE_RES_NAME,
        health_indicators::UniformData {
            health_factor,
            satiety_factor,
        },
        vec![],
    )
}

pub fn hud_popup(background_color: Color) -> ContainerBackground {
    ContainerBackground::new_raw(
        hud_popup::MATERIAL_PIPE_RES_NAME,
        hud_popup::UniformData {
            background_color: background_color.into_raw_linear(),
        },
        vec![],
    )
}

pub fn item_slot(background_color: Color) -> ContainerBackground {
    ContainerBackground::new_raw(
        item_slot::MATERIAL_PIPE_RES_NAME,
        item_slot::UniformData {
            background_color: background_color.into_raw_linear(),
        },
        vec![],
    )
}

pub fn material_item(image: ImageSource) -> ContainerBackground {
    ContainerBackground::new_raw(
        material_item::MATERIAL_PIPE_RES_NAME,
        material_item::UniformData {
            // background_color: background_color.into_raw_linear(),
        },
        vec![image.to_gpu_resource("material_item").into()],
    )
}

pub fn register(ctx: &EngineContext) {
    fancy::register(ctx);
    game_effects::register(ctx);
    health_indicators::register(ctx);
    hud_popup::register(ctx);
    item_slot::register(ctx);
    material_item::register(ctx);
}

pub const DEFAULT_FANCY_COLOR: Color = Color::rgb(0.4, 0.5, 0.4);
