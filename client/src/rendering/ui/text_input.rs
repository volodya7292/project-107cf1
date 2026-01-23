use crate::rendering::ui::container::{container, container_props, container_props_init};
use crate::rendering::ui::text::reactive::{ui_text, ui_text_props};
use crate::rendering::ui::{container, ui_callbacks};
use clipboard::ClipboardProvider;
use common::glm::Vec2;
use common::make_static_id;
use common::utils::StringExt;
use engine::ecs::component::simple_text::TextStyle;
use engine::ecs::component::ui::{Constraint, Position, Sizing, UILayoutC, UILayoutCacheC, UITransform};
use engine::event::WSIKeyboardInput;
use engine::module::input::Input;
use engine::module::scene::Scene;
use engine::module::text_renderer::TextRenderer;
use engine::module::ui::color::Color;
use engine::module::ui::reactive::{ReactiveState, UIScopeContext};
use engine::utils::transition::AnimatedValue;
use engine::utils::transition::TransitionTarget;
use engine::winit::event::ElementState;
use engine::winit::keyboard::KeyCode;
use engine::{remember_state, EngineContext};
use entity_data::EntityId;
use std::sync::Arc;

#[derive(Default, Clone, PartialEq)]
pub struct TextInputProps {
    pub layout: UILayoutC,
    pub multiline: bool,
    pub text_state: ReactiveState<String>,
    pub style: TextStyle,
}

fn cursor_blink_time_fn(_: f32) -> f32 {
    1.0
}

pub fn ui_text_input(local_name: &str, ctx: &mut UIScopeContext, props: TextInputProps) {
    container(
        local_name,
        ctx,
        container_props_init(props).layout(UILayoutC::new().with_width_grow()),
        move |ctx, props| {
            let line_height = {
                let text_renderer = ctx.ctx().module::<TextRenderer>();
                text_renderer.get_min_line_height(&props.style)
            };

            remember_state!(ctx, cursor_pos, 0);
            remember_state!(ctx, text_offset, Vec2::new(0.0, 0.0));
            remember_state!(ctx, text_size, Vec2::new(0.0, 0.0));
            remember_state!(ctx, text_global_pos, Vec2::new(0.0, 0.0));
            remember_state!(ctx, container_size, None::<Vec2>);
            remember_state!(ctx, focused, false);

            let text = ctx.subscribe(&props.text_state);
            let n_chars = text.chars().count();

            let cursor_opacity_state =
                ctx.request_state(make_static_id!(), || AnimatedValue::immediate(1.0_f32));

            if *cursor_pos > n_chars {
                cursor_pos.state().update(n_chars);
            }

            let cursor_offset = if n_chars > 0 {
                let text_renderer = ctx.ctx().module::<TextRenderer>();
                let info = text_renderer.calculate_char_offset(
                    &text,
                    &props.style,
                    if props.multiline {
                        container_size.map_or(0.0, |v| v.x)
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
            if let Some(container_size) = *container_size {
                let text_offset_x_min = (-text_size.x + container_size.x).min(0.0);

                if relative_cursor_offset.x > container_size.x {
                    text_offset
                        .state()
                        .update_with(move |prev| Vec2::new(-cursor_offset.x + container_size.x, prev.y));
                } else if relative_cursor_offset.x < 0.0 {
                    text_offset
                        .state()
                        .update_with(move |prev| Vec2::new((-cursor_offset.x).min(0.0), prev.y));
                }

                if text_offset.x < text_offset_x_min {
                    text_offset
                        .state()
                        .update_with(move |prev| Vec2::new(text_offset_x_min, prev.y));
                }
            }

            let container_size_state = container_size.state();
            let text_size_state = text_size.state();
            let text_global_pos_state = text_global_pos.state();
            let on_text_size_update = move |entity: &EntityId, ctx: &EngineContext, _new_size: Vec2| {
                let mut scene = ctx.module_mut::<Scene>();
                let entry = scene.entry(entity);
                let layout = entry.get::<UILayoutC>();
                let layout_padding = layout.padding;
                let cache = entry.get::<UILayoutCacheC>();
                let global_pos = *cache.global_position();
                let size = *cache.final_size();
                let clip_rect = *cache.clip_rect();

                text_global_pos_state.update(global_pos + Vec2::new(layout_padding.left, layout_padding.top));
                text_size_state.update(size - layout_padding.size());
                container_size_state.update(Some(clip_rect.size() - layout_padding.size()));
            };

            let container_size_state = container_size.state();
            let text_global_pos_state = text_global_pos.state();
            let cursor_pos_state = cursor_pos.state();
            let text2 = text.clone();
            let props2 = props.clone();
            let on_click = move |_: &EntityId, ctx: &EngineContext, pos: Vec2| {
                let text_renderer = ctx.module::<TextRenderer>();

                let char_idx = text_renderer.find_nearest_char_index(
                    &text2,
                    &props2.style,
                    if props2.multiline {
                        container_size_state.value().map_or(0.0, |v| v.x)
                    } else {
                        f32::INFINITY
                    },
                    pos - *text_global_pos_state.value(),
                );

                if let Some(char_idx) = char_idx {
                    cursor_pos_state.update(char_idx);
                }
            };

            let focused_state = focused.state();
            let on_focus_in = move |_: &EntityId, _: &EngineContext| {
                focused_state.update(true);
            };

            let focused_state = focused.state();
            let on_focus_out = move |_: &EntityId, _: &EngineContext| {
                focused_state.update(false);
            };

            let cursor_pos_state = cursor_pos.state();
            let cursor_opacity_state2 = cursor_opacity_state.clone();
            let text_state = text.state();
            let on_key_press = move |_: &EntityId, ctx: &EngineContext, input: WSIKeyboardInput| {
                let super_key_pressed = {
                    let input_manager = ctx.module::<Input>();
                    input_manager.keyboard().is_super_key_pressed()
                };

                let cursor_pos_state = cursor_pos_state.clone();
                let cursor_opacity_state = cursor_opacity_state2.clone();
                let text_state = text_state.clone();

                text_state.update_with(move |prev| {
                    let mut new_text = prev.clone();
                    let mut new_cursor_offset = 0_isize;

                    if let WSIKeyboardInput::Virtual(keycode, state, ch) = input {
                        if let Some(keycode) = keycode && state == ElementState::Pressed {
                            if keycode == KeyCode::ArrowLeft {
                                new_cursor_offset = -1;
                            } else if keycode == KeyCode::ArrowRight {
                                new_cursor_offset = 1;
                            } else if keycode == KeyCode::Backspace {
                                if *cursor_pos_state.value() > 0 {
                                    new_text.remove_at_char(*cursor_pos_state.value() - 1);
                                    new_cursor_offset = -1;
                                }
                            } else if keycode == KeyCode::Delete {
                                let cursor_pos = *cursor_pos_state.value();
                                if new_text.chars().count() > cursor_pos {
                                    new_text.remove_at_char(cursor_pos);
                                }
                            }

                            if super_key_pressed && keycode == KeyCode::KeyV {
                                let contents = clipboard::ClipboardContext::new()
                                    .unwrap()
                                    .get_contents()
                                    .unwrap_or_default();
                                new_text.insert_str_at_char(*cursor_pos_state.value(), &contents);
                                new_cursor_offset = contents.chars().count() as isize;
                            }
                        }

                        if let Some(ch) = ch {
                            if ch.is_control() || super_key_pressed {
                                return new_text;
                            }
                            new_text.insert_at_char(*cursor_pos_state.value(), ch);
                            new_cursor_offset = 1;
                        }
                    }

                    let new_n_chars = new_text.chars().count();
                    cursor_pos_state.update_with(move |prev| {
                        (*prev as isize + new_cursor_offset).clamp(0, new_n_chars as isize) as usize
                    });

                    new_text
                });

                cursor_opacity_state.update_with(|prev| {
                    let mut new = *prev;
                    new.retarget((*prev.target()).with_value(1.0));
                    new
                });
            };

            container(
                make_static_id!(),
                ctx,
                container_props_init((
                    text.clone(),
                    props.style,
                    props.multiline,
                    cursor_offset,
                    line_height,
                    *focused,
                ))
                .layout(
                    props
                        .layout
                        .with_content_transform(UITransform::new().with_offset(*text_offset))
                        .with_min_height(line_height)
                        .with_grow(),
                )
                .callbacks(
                    ui_callbacks()
                        .focusable(true)
                        .on_click(Arc::new(on_click))
                        .on_focus_in(Arc::new(on_focus_in))
                        .on_focus_out(Arc::new(on_focus_out))
                        .on_keyboard(Arc::new(on_key_press))
                        .on_size_update(Arc::new(on_text_size_update)),
                ),
                move |ctx, (text, style, multiline, cursor_offset, line_height, focused)| {
                    ui_text(
                        make_static_id!(),
                        ctx,
                        ui_text_props(text)
                            .layout(
                                UILayoutC::new()
                                    .with_width_grow()
                                    .with_height(Sizing::FitContent)
                                    .with_min_height(line_height),
                            )
                            .callbacks(ui_callbacks().interaction(false))
                            .style(style)
                            .wrap(multiline),
                    );
                    if focused {
                        let cursor_opacity = ctx.subscribe(&cursor_opacity_state);
                        if cursor_opacity.is_finished() {
                            cursor_opacity_state.update_with(|prev| {
                                let mut new = *prev;
                                new.retarget(
                                    TransitionTarget::new(1.0 - *prev.target().value(), 0.5)
                                        .with_time_fn(cursor_blink_time_fn),
                                );
                                new
                            });
                        }
                        ctx.drive_transition(&cursor_opacity);

                        container(
                            make_static_id!(),
                            ctx,
                            container_props()
                                .layout(
                                    UILayoutC::new()
                                        .with_position(Position::Relative(cursor_offset))
                                        .with_width_constraint(Constraint::exact(1.0))
                                        .with_height_constraint(Constraint::exact(line_height)),
                                )
                                .background(Some(container::background::solid_color(Color::grayscale(0.5))))
                                .opacity(*cursor_opacity.current()),
                            |_, ()| {},
                        );
                    }
                },
            );
        },
    );
}
