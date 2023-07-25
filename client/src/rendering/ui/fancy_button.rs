use crate::game::EngineCtxGameExt;
use crate::rendering::ui::container::ContainerBackground;
use crate::rendering::ui::container::{container, ContainerProps};
use crate::rendering::ui::text::reactive::{ui_text, UITextProps};
use crate::rendering::ui::text::UITextImpl;
use crate::rendering::ui::UICallbacks;
use common::glm::Vec4;
use common::make_static_id;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{ClickedCallback, UILayoutC};
use engine::module::main_renderer::MainRenderer;
use engine::module::scene::Scene;
use engine::module::ui::color::Color;
use engine::module::ui::reactive::UIScopeContext;
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::vkw::pipeline::CullMode;
use engine::vkw::PrimitiveTopology;
use engine::EngineContext;
use entity_data::EntityId;
use std::sync::Arc;

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct UniformData {
    background_color: Vec4,
}

const DEFAULT_NORMAL_COLOR: Color = Color::grayscale(0.2);
const MATERIAL_PIPE_RES_NAME: &str = make_static_id!();

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
            include_bytes!("../../../res/shaders/fancy_button.frag.spv"),
            "fancy_button.frag",
        )
        .unwrap();

    let mat_pipe_id = renderer.register_material_pipeline(
        &[vertex, pixel],
        PrimitiveTopology::TRIANGLE_STRIP,
        CullMode::BACK,
    );

    scene.register_named_resource(MATERIAL_PIPE_RES_NAME, mat_pipe_id);
}

pub fn fancy_button(
    local_id: &str,
    ctx: &mut UIScopeContext,
    layout: UILayoutC,
    text: StyledString,
    on_click: impl ClickedCallback,
) {
    let on_click = Arc::new(on_click);

    let uniform_data = UniformData {
        background_color: DEFAULT_NORMAL_COLOR.into_raw(),
    };

    let text_color = ctx.request_state(format!("{}_{}", local_id, "text_color"), || {
        AnimatedValue::immediate(*text.style().color())
    });
    let curr_text_color = *text.style().color();

    let text_color2 = text_color.clone();
    let on_cursor_enter = Arc::new(move |_: &EntityId, ctx: &EngineContext| {
        let app = ctx.app();
        let mut reactor = app.ui_reactor();

        const MUL_FACTOR: f32 = 3.0;
        let mut active_color = curr_text_color.into_raw();
        active_color.x *= MUL_FACTOR;
        active_color.y *= MUL_FACTOR;
        active_color.z *= MUL_FACTOR;

        reactor.set_state(&text_color2, |prev| {
            let mut new = prev.clone();
            new.retarget(TransitionTarget::new(active_color.into(), 0.2));
            new
        });
    });

    let text_color2 = text_color.clone();
    let on_cursor_leave = Arc::new(move |_: &EntityId, ctx: &EngineContext| {
        let app = ctx.app();
        let mut reactor = app.ui_reactor();

        reactor.set_state(&text_color2, |prev| {
            let mut new = prev.clone();
            new.retarget(TransitionTarget::new(curr_text_color, 0.2));
            new
        });
    });

    container(
        local_id,
        ctx,
        ContainerProps {
            layout,
            background: Some(ContainerBackground::new_raw(MATERIAL_PIPE_RES_NAME, uniform_data)),
            callbacks: UICallbacks::new()
                .with_on_click(on_click.clone())
                .with_on_cursor_enter(on_cursor_enter)
                .with_on_cursor_leave(on_cursor_leave),
            ..Default::default()
        },
        move |ctx| {
            let text_color = ctx.subscribe(&text_color);

            ctx.drive_transition(&text_color);

            ui_text(
                make_static_id!(),
                ctx,
                UITextProps {
                    text: text.data().to_string(),
                    style: text.style().clone().with_color(*text_color.current()),
                    ..Default::default()
                },
            );
        },
    );
}
