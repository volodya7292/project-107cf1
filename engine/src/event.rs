use crate::utils;
use crate::utils::wsi::WSIPosition;
use nalgebra_glm::Vec2;
use winit::window::Window;

pub enum Event {
    WindowEvent { event: WindowEvent },
}

pub enum WindowEvent {
    CursorMoved { position: WSIPosition<f32> },
}

pub(crate) type WEvent<'a> = winit::event::Event<'a, ()>;
pub(crate) type WWindowEvent<'a> = winit::event::WindowEvent<'a>;

impl Event {
    pub(crate) fn from_winit(winit_event: &WEvent, window: &Window) -> Option<Event> {
        let event = match winit_event {
            WEvent::WindowEvent { event: win_event, .. } => Event::WindowEvent {
                event: match win_event {
                    WWindowEvent::CursorMoved { position, .. } => WindowEvent::CursorMoved {
                        position: WSIPosition::<f32>::from_winit(
                            (position.x as f32, position.y as f32),
                            window,
                        ),
                    },
                    _ => return None,
                },
            },
            _ => return None,
        };
        Some(event)
    }
}
