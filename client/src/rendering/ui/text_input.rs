use crate::rendering::ui::container::{container, ContainerProps};
use crate::rendering::ui::text::reactive::{ui_text, UITextProps};
use crate::rendering::ui::{container, UICallbacks};
use clipboard::ClipboardProvider;
use common::glm::Vec2;
use common::make_static_id;
use common::utils::StringExt;
use engine::ecs::component::simple_text::TextStyle;
use engine::ecs::component::ui::{
    Constraint, Position, Sizing, UILayoutC, UILayoutCacheC, UITransform, Visibility,
};
use engine::event::WSIKeyboardInput;
use engine::module::input::Input;
use engine::module::scene::Scene;
use engine::module::text_renderer::TextRenderer;
use engine::module::ui::color::Color;
use engine::module::ui::reactive::{ReactiveState, UIScopeContext};
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
    pub text_state: ReactiveState<String>,
    pub style: TextStyle,
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
            layout: UILayoutC::new().with_width_grow(),
            ..Default::default()
        },
        move |ctx| {
            let line_height = {
                let text_renderer = ctx.ctx().module::<TextRenderer>();
                text_renderer.get_min_line_height(&props.style)
            };

            remember_state!(ctx, cursor_pos, 1);
            remember_state!(ctx, cursor_opacity, AnimatedValue::immediate(1.0));
            remember_state!(ctx, text_offset, Vec2::new(0.0, 0.0));
            remember_state!(ctx, text_size, Vec2::new(0.0, 0.0));
            remember_state!(ctx, text_global_pos, Vec2::new(0.0, 0.0));
            remember_state!(ctx, container_size, None::<Vec2>);
            remember_state!(ctx, focused, false);

            let container_size_state = container_size.state().clone();
            let text_size_state = text_size.state().clone();
            let text_global_pos_state = text_global_pos.state().clone();
            let cursor_opacity_state = cursor_opacity.state().clone();

            let text = ctx.subscribe(&props.text_state);
            let n_chars = text.chars().count();

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

            if *cursor_pos > n_chars {
                cursor_pos.state().update(n_chars);
            }

            let cursor_offset = if n_chars > 0 {
                let text_renderer = ctx.ctx().module::<TextRenderer>();
                let info = text_renderer.calculate_char_offset(
                    &*text,
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
                let text_offset_x_min = (-text_size.x.max(container_size.x - CURSOR_X_PADDING)
                    + (container_size.x - CURSOR_X_PADDING));

                if relative_cursor_offset.x > container_size.x - CURSOR_X_PADDING {
                    text_offset.state().update_with(move |prev| {
                        Vec2::new(-cursor_offset.x + container_size.x - CURSOR_X_PADDING, prev.y)
                    });
                } else if relative_cursor_offset.x < CURSOR_X_PADDING {
                    text_offset.state().update_with(move |prev| {
                        Vec2::new((-cursor_offset.x + CURSOR_X_PADDING).min(0.0), prev.y)
                    });
                }
                if text_offset.x < text_offset_x_min {
                    text_offset
                        .state()
                        .update_with(move |prev| Vec2::new(text_offset_x_min, prev.y));
                }
            }

            let on_text_size_update = move |entity: &EntityId, ctx: &EngineContext| {
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

            let container_size_state = container_size.state().clone();
            let text_global_pos_state = text_global_pos.state().clone();
            let cursor_pos_state = cursor_pos.state().clone();
            let text2 = text.clone();
            let props2 = props.clone();
            let on_click = move |entity: &EntityId, ctx: &EngineContext, pos: Vec2| {
                let text_renderer = ctx.module::<TextRenderer>();

                let char_idx = text_renderer.find_char_index(
                    &text2,
                    &props2.style,
                    if props2.multiline {
                        container_size_state.value().map_or(0.0, |v| v.x)
                    } else {
                        f32::INFINITY
                    },
                    pos - text_global_pos_state.value(),
                );

                if let Some(char_idx) = char_idx {
                    cursor_pos_state.update(char_idx);
                }
            };

            let focused_state = focused.state().clone();
            let on_focus_in = move |entity: &EntityId, ctx: &EngineContext| {
                focused_state.update(true);
            };

            let focused_state = focused.state().clone();
            let on_focus_out = move |entity: &EntityId, ctx: &EngineContext| {
                focused_state.update(false);
            };

            let cursor_pos_state = cursor_pos.state().clone();
            let cursor_opacity_state = cursor_opacity.state().clone();
            let text_state = text.state().clone();
            let on_key_press = move |entity: &EntityId, ctx: &EngineContext, input: WSIKeyboardInput| {
                let super_key_pressed = {
                    let input_manager = ctx.module::<Input>();
                    input_manager.keyboard().is_super_key_pressed()
                };

                let cursor_pos_state = cursor_pos_state.clone();
                let text_state = text_state.clone();

                text_state.update_with(move |prev| {
                    let mut new_text = prev.clone();
                    let mut new_cursor_offset = 0_isize;

                    if let WSIKeyboardInput::Virtual(keycode, state) = input {
                        if state == ElementState::Pressed {
                            if keycode == VirtualKeyCode::Left {
                                new_cursor_offset = -1;
                            } else if keycode == VirtualKeyCode::Right {
                                new_cursor_offset = 1;
                            } else if keycode == VirtualKeyCode::Back {
                                if *cursor_pos_state.value() > 0 {
                                    new_text.remove_at_char(*cursor_pos_state.value() - 1);
                                    new_cursor_offset = -1;
                                }
                            } else if keycode == VirtualKeyCode::Delete {
                                let cursor_pos = *cursor_pos_state.value();
                                if new_text.chars().count() > cursor_pos {
                                    new_text.remove_at_char(cursor_pos);
                                }
                            }

                            if super_key_pressed && keycode == VirtualKeyCode::V {
                                let contents = clipboard::ClipboardContext::new()
                                    .unwrap()
                                    .get_contents()
                                    .unwrap_or_default();
                                new_text.insert_str_at_char(*cursor_pos_state.value(), &contents);
                                new_cursor_offset = contents.chars().count() as isize;
                            }
                        }
                    }
                    if let WSIKeyboardInput::Char(ch) = input {
                        if ch.is_control() || super_key_pressed {
                            return new_text;
                        }
                        new_text.insert_at_char(*cursor_pos_state.value(), ch);
                        new_cursor_offset = 1;
                    }

                    cursor_pos_state
                        .update_with(move |prev| (*prev as isize + new_cursor_offset).max(0) as usize);

                    new_text
                });

                cursor_opacity_state.update_with(|prev| {
                    let mut new = *prev;
                    new.retarget(prev.target().clone().with_value(1.0));
                    new
                });
            };

            container(
                make_static_id!(),
                ctx,
                ContainerProps {
                    layout: props
                        .layout
                        .with_content_transform(UITransform::new().with_offset(*text_offset))
                        .with_min_height(line_height)
                        .with_grow(),
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
                            layout: UILayoutC::new()
                                .with_width_grow()
                                .with_height(Sizing::FitContent)
                                .with_min_height(line_height),
                            callbacks: UICallbacks::new().with_enabled(false),
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
                                    .with_height_constraint(Constraint::exact(line_height)),
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
