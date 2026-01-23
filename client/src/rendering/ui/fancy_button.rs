use super::container::container_props_init;
use crate::rendering::ui::backgrounds::DEFAULT_FANCY_COLOR;
use crate::rendering::ui::container::container;
use crate::rendering::ui::text::reactive::{ui_text, ui_text_props};
use crate::rendering::ui::{backgrounds, ui_callbacks};
use common::make_static_id;
use engine::ecs::component::simple_text::StyledString;
use engine::ecs::component::ui::{ClickedCallback, Padding, UILayoutC};
use engine::module::ui::reactive::UIScopeContext;
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::{EngineContext, remember_state};
use entity_data::EntityId;
use std::sync::Arc;

pub fn fancy_button(
    local_id: &str,
    ctx: &mut UIScopeContext,
    layout: UILayoutC,
    text: StyledString,
    on_click: ClickedCallback,
) {
    container(
        local_id,
        ctx,
        container_props_init((layout, text)).layout(layout.with_padding(Padding::ZERO)),
        move |ctx, (layout, text)| {
            remember_state!(ctx, text_color, AnimatedValue::immediate(*text.style().color()));
            let curr_text_color = *text.style().color();

            let text_color2 = text_color.state();
            let on_cursor_enter = Arc::new(move |_: &EntityId, _: &EngineContext| {
                const MUL_FACTOR: f32 = 3.0;
                let mut active_color = curr_text_color.into_raw_linear();
                active_color.x *= MUL_FACTOR;
                active_color.y *= MUL_FACTOR;
                active_color.z *= MUL_FACTOR;

                text_color2.update_with(move |prev| {
                    let mut new = *prev;
                    new.retarget(TransitionTarget::new(active_color.into(), 0.2));
                    new
                });
            });

            let text_color2 = text_color.state();
            let on_cursor_leave = Arc::new(move |_: &EntityId, _: &EngineContext| {
                text_color2.update_with(move |prev| {
                    let mut new = *prev;
                    new.retarget(TransitionTarget::new(curr_text_color, 0.2));
                    new
                });
            });

            ctx.drive_transition(&text_color);

            container(
                make_static_id!(),
                ctx,
                container_props_init((text.clone(), *text_color))
                    .layout(layout.with_grow())
                    .background(Some(backgrounds::fancy(DEFAULT_FANCY_COLOR)))
                    .callbacks(
                        ui_callbacks()
                            .on_click(on_click.clone())
                            .on_cursor_enter(on_cursor_enter)
                            .on_cursor_leave(on_cursor_leave),
                    ),
                move |ctx, (text, text_color)| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        ui_text_props(text.data().to_string())
                            .callbacks(ui_callbacks().interaction(false))
                            .style(text.style().with_color(*text_color.current())),
                    );
                },
            );
        },
    );
}
