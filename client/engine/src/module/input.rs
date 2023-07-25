use crate::event::{WSIEvent, WSIKeyboardInput};
use crate::module::EngineModule;
use crate::EngineContext;
use common::types::HashSet;
use winit::event::{ElementState, MouseButton, VirtualKeyCode};
use winit::window::Window;

pub struct Input {
    keyboard: Keyboard,
    mouse: Mouse,
}

pub struct Keyboard {
    pub(crate) pressed_keys: HashSet<VirtualKeyCode>,
}

pub struct Mouse {
    pub(crate) pressed_buttons: HashSet<MouseButton>,
}

impl Keyboard {
    pub(crate) fn new() -> Self {
        Self {
            pressed_keys: HashSet::with_capacity(16),
        }
    }
    pub fn is_key_pressed(&self, keycode: VirtualKeyCode) -> bool {
        self.pressed_keys.contains(&keycode)
    }
}

impl Mouse {
    pub(crate) fn new() -> Self {
        Self {
            pressed_buttons: HashSet::with_capacity(8),
        }
    }

    pub fn is_button_pressed(&self, button: MouseButton) -> bool {
        self.pressed_buttons.contains(&button)
    }
}

impl Input {
    pub fn new() -> Self {
        Self {
            keyboard: Keyboard::new(),
            mouse: Mouse::new(),
        }
    }

    pub fn mouse(&self) -> &Mouse {
        &self.mouse
    }

    pub fn keyboard(&self) -> &Keyboard {
        &self.keyboard
    }
}

impl EngineModule for Input {
    fn on_wsi_event(&mut self, _: &Window, event: &WSIEvent, _: &EngineContext) {
        match event {
            WSIEvent::KeyboardInput { input, .. } => {
                if let WSIKeyboardInput::Virtual(keycode, state) = input {
                    if *state == ElementState::Pressed {
                        self.keyboard.pressed_keys.insert(*keycode);
                    } else {
                        self.keyboard.pressed_keys.remove(keycode);
                    }
                }
            }
            WSIEvent::MouseInput { state, button, .. } => {
                if *state == ElementState::Pressed {
                    self.mouse.pressed_buttons.insert(*button);
                } else {
                    self.mouse.pressed_buttons.remove(button);
                }
            }
            _ => {}
        }
    }
}
