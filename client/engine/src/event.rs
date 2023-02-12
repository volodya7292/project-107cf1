use crate::utils::wsi::{WSIPosition, WSISize};
use winit::event::{ElementState, MouseButton};
use winit::window::Window;

pub enum WSIEvent {
    Resized(WSISize<u32>),
    CursorMoved {
        position: WSIPosition<f32>,
    },
    MouseInput {
        state: ElementState,
        button: MouseButton,
    },
}

pub(crate) type WEvent<'a> = winit::event::Event<'a, ()>;
pub(crate) type WWindowEvent<'a> = winit::event::WindowEvent<'a>;

impl WSIEvent {
    pub(crate) fn from_winit(winit_event: &WEvent, window: &Window) -> Option<WSIEvent> {
        let WEvent::WindowEvent { event: win_event, window_id, ..} = winit_event else {
            return None;
        };
        if *window_id != window.id() {
            return None;
        }

        let wsi_event = match win_event {
            WWindowEvent::Resized(_) | WWindowEvent::ScaleFactorChanged { .. } => {
                let raw_size = window.inner_size();
                if raw_size.width == 0 || raw_size.height == 0 {
                    return None;
                }
                let new_wsi_size = WSISize::<u32>::from_winit((raw_size.width, raw_size.height), window);
                WSIEvent::Resized(new_wsi_size)
            }
            WWindowEvent::CursorMoved { position, .. } => WSIEvent::CursorMoved {
                position: WSIPosition::<f32>::from_winit((position.x as f32, position.y as f32), window),
            },
            WWindowEvent::MouseInput { state, button, .. } => WSIEvent::MouseInput {
                state: *state,
                button: *button,
            },
            _ => return None,
        };

        Some(wsi_event)
    }
}
