use crate::game::EngineCtxGameExt;
use crate::rendering::ui::container::{container, ContainerProps};
use crate::rendering::ui::text::reactive::{ui_text, UITextProps};
use crate::rendering::ui::{container, UICallbacks};
use common::glm::Vec2;
use common::make_static_id;
use common::utils::StringExt;
use engine::ecs::component::simple_text::TextStyle;
use engine::ecs::component::ui::{Constraint, Position, UILayoutC, UILayoutCacheC, UITransform, Visibility};
use engine::event::WSIKeyboardInput;
use engine::module::scene::Scene;
use engine::module::text_renderer::TextRenderer;
use engine::module::ui::color::Color;
use engine::module::ui::reactive::UIScopeContext;
use engine::utils::transition::AnimatedValue;
use engine::utils::transition::TransitionTarget;
use engine::winit::event::{ElementState, VirtualKeyCode};
use engine::{define_callback, remember_state, EngineContext};
use entity_data::EntityId;
use std::sync::Arc;

define_callback!(TextChangeCallback(&EngineContext, String));

#[derive(Default, Clone)]
pub struct TextInputProps {
    pub layout: UILayoutC,
    pub multiline: bool,
    pub initial_text: String,
    pub style: TextStyle,
    pub on_change: Option<Arc<dyn TextChangeCallback<Output = ()>>>,
}

const CURSOR_X_PADDING: f32 = 10.0;

fn cursor_blink_time_fn(_: f32) -> f32 {
    1.0
}

pub fn ui_text_input(local_name: &str, ctx: &mut UIScopeContext, props: TextInputProps) {
    container(
        local_name,
        ctx,
        ContainerProps {
            layout: props.layout,
            ..Default::default()
        },
        move |ctx| {
            let char_height = props.style.font_size();

            remember_state!(ctx, text, props.initial_text.clone());
            remember_state!(ctx, cursor_pos, 1);
            remember_state!(ctx, cursor_opacity, AnimatedValue::immediate(1.0));
            remember_state!(ctx, text_offset, Vec2::new(-100.0, 0.0));
            remember_state!(ctx, text_size, Vec2::new(0.0, 0.0));
            remember_state!(ctx, text_global_pos, Vec2::new(0.0, 0.0));
            remember_state!(ctx, container_size, Vec2::new(0.0, 0.0));
            remember_state!(ctx, focused, false);

            let cursor_pos_state = cursor_pos.state().clone();
            let container_size_state = container_size.state().clone();
            let text_size_state = text_size.state().clone();
            let text_global_pos_state = text_global_pos.state().clone();
            let cursor_opacity_state = cursor_opacity.state().clone();

            let n_chars = text.chars().count();

            if cursor_opacity.is_finished() {
                ctx.reactor().set_state(&cursor_opacity_state, |prev| {
                    let mut new = *prev;
                    new.retarget(
                        TransitionTarget::new(1.0 - *prev.target().value(), 0.5)
                            .with_time_fn(cursor_blink_time_fn),
                    );
                    new
                });
            }
            ctx.drive_transition(&cursor_opacity);

            if *cursor_pos > n_chars {
                ctx.reactor().set_state(cursor_pos.state(), |prev| n_chars);
            }

            let cursor_offset = if n_chars > 0 {
                let text_renderer = ctx.ctx().module::<TextRenderer>();
                let info = text_renderer.calculate_char_offset(
                    &*text,
                    &props.style,
                    if props.multiline {
                        container_size.x
                    } else {
                        f32::INFINITY
                    },
                    (*cursor_pos).min(n_chars - 1),
                );

                let mut offset = *info.offset();
                if *cursor_pos == n_chars {
                    offset.x += info.size().x;
                }

                offset
            } else {
                Default::default()
            };

            let relative_cursor_offset = cursor_offset + *text_offset;
            let text_offset_x_min = (-text_size.x.max(container_size.x - CURSOR_X_PADDING)
                + (container_size.x - CURSOR_X_PADDING));

            if relative_cursor_offset.x > container_size.x - CURSOR_X_PADDING {
                ctx.reactor().set_state(text_offset.state(), |prev| {
                    Vec2::new(-cursor_offset.x + container_size.x - CURSOR_X_PADDING, prev.y)
                });
            } else if relative_cursor_offset.x < CURSOR_X_PADDING {
                ctx.reactor().set_state(text_offset.state(), |prev| {
                    Vec2::new((-cursor_offset.x + CURSOR_X_PADDING).min(0.0), prev.y)
                });
            }
            if text_offset.x < text_offset_x_min {
                ctx.reactor()
                    .set_state(text_offset.state(), |prev| Vec2::new(text_offset_x_min, prev.y));
            }

            let on_text_size_update = move |entity: &EntityId, ctx: &EngineContext| {
                let app = ctx.app();
                let mut reactor = app.ui_reactor();
                let mut scene = ctx.module_mut::<Scene>();
                let entry = scene.entry(entity);
                let cache = entry.get::<UILayoutCacheC>();
                let global_pos = *cache.global_position();
                let size = *cache.final_size();
                let clip_rect = *cache.clip_rect();

                reactor.set_state(&text_global_pos_state.clone(), |_| global_pos);
                reactor.set_state(&text_size_state.clone(), |_| size);
                reactor.set_state(&container_size_state.clone(), |_| clip_rect.size());
            };

            let container_size_state = container_size.state().clone();
            let text_global_pos_state = text_global_pos.state().clone();
            let text2 = text.clone();
            let props2 = props.clone();
            let on_click = move |entity: &EntityId, ctx: &EngineContext, pos: Vec2| {
                let text_renderer = ctx.module::<TextRenderer>();
                let char_idx = text_renderer.find_char_index(
                    &text2,
                    &props2.style,
                    if props2.multiline {
                        container_size_state.value().x
                    } else {
                        f32::INFINITY
                    },
                    pos - text_global_pos_state.value(),
                );

                if let Some(char_idx) = char_idx {
                    let app = ctx.app();
                    let mut reactor = app.ui_reactor();
                    reactor.set_state(&cursor_pos_state, |_| char_idx);
                }
            };

            let focused_state = focused.state().clone();
            let on_focus_in = move |entity: &EntityId, ctx: &EngineContext| {
                let app = ctx.app();
                let mut reactor = app.ui_reactor();
                reactor.set_state(&focused_state.clone(), |_| true);
            };

            let focused_state = focused.state().clone();
            let on_focus_out = move |entity: &EntityId, ctx: &EngineContext| {
                let app = ctx.app();
                let mut reactor = app.ui_reactor();
                reactor.set_state(&focused_state.clone(), |_| false);
            };

            let cursor_pos_state = cursor_pos.state().clone();
            let cursor_opacity_state = cursor_opacity.state().clone();
            let text_state = text.state().clone();
            let on_change = props.on_change.clone();
            let on_key_press = move |entity: &EntityId, ctx: &EngineContext, input: WSIKeyboardInput| {
                let mut new_cursor_offset = 0_isize;
                let mut new_text = text_state.value().clone();

                match input {
                    WSIKeyboardInput::Virtual(keycode, state) => {
                        if state == ElementState::Pressed {
                            if keycode == VirtualKeyCode::Left {
                                new_cursor_offset = -1;
                            } else if keycode == VirtualKeyCode::Right {
                                new_cursor_offset = 1;
                            } else if keycode == VirtualKeyCode::Back {
                                if *cursor_pos_state.value() > 0 {
                                    new_text.remove_char(*cursor_pos_state.value() - 1);
                                    new_cursor_offset = -1;
                                }
                            } else if keycode == VirtualKeyCode::Delete {
                                let cursor_pos = *cursor_pos_state.value();
                                if new_text.chars().count() > cursor_pos {
                                    new_text.remove_char(cursor_pos);
                                }
                            }
                        }
                    }
                    WSIKeyboardInput::Char(ch) => {
                        if ch.is_control() {
                            return;
                        }
                        new_text.insert_char(*cursor_pos_state.value(), ch);
                        new_cursor_offset = 1;
                    }
                }

                if &new_text != text_state.value() {
                    if let Some(on_change) = &on_change {
                        on_change(ctx, new_text.clone());
                    }
                }

                let app = ctx.app();
                let mut reactor = app.ui_reactor();
                reactor.set_state(&cursor_pos_state, |prev| {
                    (*prev as isize + new_cursor_offset).max(0) as usize
                });
                reactor.set_state(&cursor_opacity_state, |prev| {
                    let mut new = *prev;
                    new.retarget(prev.target().clone().with_value(1.0));
                    new
                });
                reactor.set_state(&text_state, |prev| new_text);
            };

            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: UILayoutC::new()
                        .with_content_transform(UITransform::new().with_offset(*text_offset))
                        .with_grow(),
                    // .with_min_width(800.0),
                    callbacks: UICallbacks::new()
                        .with_focusable(true)
                        .with_on_click(Arc::new(on_click))
                        .with_on_focus_in(Arc::new(on_focus_in))
                        .with_on_focus_out(Arc::new(on_focus_out))
                        .with_on_key_press(Arc::new(on_key_press))
                        .with_on_size_update(Arc::new(on_text_size_update)),
                    ..Default::default()
                },
                move |ctx| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        UITextProps {
                            text: text.clone(),
                            style: props.style,
                            wrap: props.multiline,
                            ..Default::default()
                        },
                    );
                    if *focused {
                        container(
                            make_static_id!(),
                            ctx,
                            ContainerProps {
                                layout: UILayoutC::new()
                                    .with_position(Position::Relative(cursor_offset))
                                    .with_visibility(Visibility::Opacity(*cursor_opacity))
                                    .with_width_constraint(Constraint::exact(1.0))
                                    .with_height_constraint(Constraint::exact(char_height)),
                                background: Some(container::background::solid_color(Color::grayscale(0.5))),
                                ..Default::default()
                            },
                            |_| {},
                        );
                    }
                },
            );
        },
    );
}
