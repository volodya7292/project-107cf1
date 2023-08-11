use crate::utils::wsi::vec2::WSizingInfo;
use crate::utils::wsi::{WSIPosition, WSISize};
use common::glm::{DVec2, DVec3};
use winit::event::{
    DeviceEvent, ElementState, ModifiersState, MouseButton, MouseScrollDelta, VirtualKeyCode,
};
use winit::window::Window;

#[derive(Copy, Clone)]
pub enum WSIKeyboardInput {
    Virtual(VirtualKeyCode, ElementState),
    Char(char),
    Modifiers(ModifiersState),
}

pub enum WSIEvent {
    Resized(WSISize<u32>),
    CursorMoved {
        position: WSIPosition<f32>,
    },
    MouseInput {
        state: ElementState,
        button: MouseButton,
    },
    MouseMotion {
        delta: DVec2,
    },
    MouseWheel {
        delta: f64,
    },
    KeyboardInput {
        input: WSIKeyboardInput,
    },
}

pub(crate) type WEvent<'a> = winit::event::Event<'a, ()>;
pub(crate) type WWindowEvent<'a> = winit::event::WindowEvent<'a>;

impl WSIEvent {
    pub(crate) fn from_winit(
        winit_event: &WEvent,
        window: &Window,
        sizing_info: &WSizingInfo,
    ) -> Option<WSIEvent> {
        if let WEvent::WindowEvent {
            event: win_event,
            window_id,
            ..
        } = winit_event
        {
            if *window_id != window.id() {
                return None;
            }
            let ev = match win_event {
                WWindowEvent::Resized(_) | WWindowEvent::ScaleFactorChanged { .. } => {
                    let raw_size = window.inner_size();
                    if raw_size.width == 0 || raw_size.height == 0 {
                        return None;
                    }
                    let new_wsi_size =
                        WSISize::<u32>::from_raw((raw_size.width, raw_size.height), sizing_info);
                    WSIEvent::Resized(new_wsi_size)
                }
                WWindowEvent::CursorMoved { position, .. } => WSIEvent::CursorMoved {
                    position: WSIPosition::<f32>::from_raw(
                        (position.x as f32, position.y as f32),
                        sizing_info,
                    ),
                },
                WWindowEvent::MouseInput { state, button, .. } => WSIEvent::MouseInput {
                    state: *state,
                    button: *button,
                },
                WWindowEvent::KeyboardInput { input, .. } if input.virtual_keycode.is_some() => {
                    WSIEvent::KeyboardInput {
                        input: WSIKeyboardInput::Virtual(input.virtual_keycode.unwrap(), input.state),
                    }
                }
                WWindowEvent::ModifiersChanged(state) => WSIEvent::KeyboardInput {
                    input: WSIKeyboardInput::Modifiers(*state),
                },
                WWindowEvent::ReceivedCharacter(ch) => WSIEvent::KeyboardInput {
                    input: WSIKeyboardInput::Char(*ch),
                },
                WWindowEvent::MouseWheel { delta, .. } => WSIEvent::MouseWheel {
                    delta: {
                        match delta {
                            MouseScrollDelta::LineDelta(_, y) => *y as f64,
                            MouseScrollDelta::PixelDelta(delta) => {
                                WSIPosition::<f32>::from_raw((delta.x as f32, delta.y as f32), sizing_info)
                                    .logical()
                                    .y as f64
                            }
                        }
                    },
                },
                _ => return None,
            };
            return Some(ev);
        }
        if let WEvent::DeviceEvent { event: dev_event, .. } = winit_event {
            let ev = match dev_event {
                DeviceEvent::MouseMotion { delta } => WSIEvent::MouseMotion {
                    delta: DVec2::new(delta.0, delta.1),
                },
                _ => return None,
            };
            return Some(ev);
        }

        None
    }
}
