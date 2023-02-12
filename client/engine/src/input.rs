use common::types::HashSet;
use winit::event::{MouseButton, VirtualKeyCode};

pub struct Keyboard {
    pub(crate) pressed_keys: HashSet<VirtualKeyCode>,
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

pub struct Mouse {
    pub(crate) pressed_buttons: HashSet<MouseButton>,
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
