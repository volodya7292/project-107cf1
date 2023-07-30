use crate::rendering::ui::backgrounds::DEFAULT_FANCY_COLOR;
use crate::rendering::ui::container::{container, ContainerProps};
use crate::rendering::ui::text::reactive::{ui_text, UITextProps};
use crate::rendering::ui::text::UITextImpl;
use crate::rendering::ui::{backgrounds, UICallbacks};
use common::make_static_id;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{ClickedCallback, Padding, UILayoutC};
use engine::module::ui::reactive::UIScopeContext;
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::{remember_state, EngineContext};
use entity_data::EntityId;
use std::sync::Arc;

pub fn fancy_button(
    local_id: &str,
    ctx: &mut UIScopeContext,
    layout: UILayoutC,
    text: StyledString,
    on_click: impl ClickedCallback,
) {
    let on_click = Arc::new(on_click);

    container(
        local_id,
        ctx,
        ContainerProps {
            layout: layout.with_padding(Padding::ZERO),
            children_props: (layout, text),
            ..Default::default()
        },
        move |ctx, (layout, text)| {
            remember_state!(ctx, text_color, AnimatedValue::immediate(*text.style().color()));
            let curr_text_color = *text.style().color();

            let text_color2 = text_color.state().clone();
            let on_cursor_enter = Arc::new(move |_: &EntityId, ctx: &EngineContext| {
                const MUL_FACTOR: f32 = 3.0;
                let mut active_color = curr_text_color.into_raw();
                active_color.x *= MUL_FACTOR;
                active_color.y *= MUL_FACTOR;
                active_color.z *= MUL_FACTOR;

                text_color2.update_with(move |prev| {
                    let mut new = prev.clone();
                    new.retarget(TransitionTarget::new(active_color.into(), 0.2));
                    new
                });
            });

            let text_color2 = text_color.state().clone();
            let on_cursor_leave = Arc::new(move |_: &EntityId, ctx: &EngineContext| {
                text_color2.update_with(move |prev| {
                    let mut new = prev.clone();
                    new.retarget(TransitionTarget::new(curr_text_color, 0.2));
                    new
                });
            });

            ctx.drive_transition(&text_color);

            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: layout.with_grow(),
                    background: Some(backgrounds::fancy(DEFAULT_FANCY_COLOR)),
                    callbacks: UICallbacks::new()
                        .with_on_click(on_click.clone())
                        .with_on_cursor_enter(on_cursor_enter)
                        .with_on_cursor_leave(on_cursor_leave),
                    children_props: (text.clone(), *text_color),
                    ..Default::default()
                },
                move |ctx, (text, text_color)| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        UITextProps {
                            text: text.data().to_string(),
                            callbacks: UICallbacks::new().with_enabled(false),
                            style: text.style().clone().with_color(*text_color.current()),
                            ..Default::default()
                        },
                    );
                },
            );
        },
    );
}
