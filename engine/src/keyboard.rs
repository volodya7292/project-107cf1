use crate::HashSet;
use winit::event::VirtualKeyCode;

pub struct Keyboard {
    pub(crate) pressed_keys: HashSet<VirtualKeyCode>,
}

impl Keyboard {
    pub(crate) fn new() -> Self {
        Self {
            pressed_keys: Default::default(),
        }
    }
    pub fn is_key_pressed(&self, keycode: VirtualKeyCode) -> bool {
        self.pressed_keys.contains(&keycode)
    }
}
