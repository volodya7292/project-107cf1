use crate::renderer::{component, Renderer};
use nalgebra as na;
use specs::WorldExt;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

pub struct Program {
    pub(crate) renderer: Arc<Mutex<Renderer>>,

    pressed_keys: HashSet<sdl2::keyboard::Scancode>,

    cursor_rel: (i32, i32),
}

impl Program {
    const MOVEMENT_SPEED: f32 = 2.0;
    const MOUSE_SENSITIVITY: f32 = 0.005;

    pub fn init(&self) {}

    pub fn on_event(&mut self, event: sdl2::event::Event) {
        use sdl2::event::Event;
        use sdl2::event::WindowEvent;
        match event {
            Event::KeyDown {
                timestamp,
                window_id,
                keycode,
                scancode,
                keymod,
                repeat,
            } => {
                if let Some(scancode) = scancode {
                    self.pressed_keys.insert(scancode);
                }
            }
            Event::KeyUp {
                timestamp,
                window_id,
                keycode,
                scancode,
                keymod,
                repeat,
            } => {
                if let Some(scancode) = scancode {
                    self.pressed_keys.remove(&scancode);
                }
            }
            Event::MouseMotion {
                timestamp,
                window_id,
                which,
                mousestate,
                x,
                y,
                xrel,
                yrel,
            } => {
                self.cursor_rel = (xrel, yrel);
            }
            _ => {}
        }
    }

    pub fn is_key_pressed(&self, scancode: sdl2::keyboard::Scancode) -> bool {
        self.pressed_keys.contains(&scancode)
    }

    pub fn on_update(&mut self, delta_time: f64) {
        {
            use sdl2::keyboard::Scancode;

            let mut vel_front_back = 0;
            let mut vel_left_right = 0;
            let mut vel_up_down = 0;

            if self.is_key_pressed(Scancode::W) {
                vel_front_back += 1;
            }
            if self.is_key_pressed(Scancode::S) {
                vel_front_back -= 1;
            }
            if self.is_key_pressed(Scancode::A) {
                vel_left_right -= 1;
            }
            if self.is_key_pressed(Scancode::D) {
                vel_left_right += 1;
            }
            if self.is_key_pressed(Scancode::Space) {
                vel_up_down += 1;
            }
            if self.is_key_pressed(Scancode::LShift) {
                vel_up_down -= 1;
            }

            let mut renderer = self.renderer.lock().unwrap();
            let entity = renderer.get_active_camera();
            let mut camera_comp = renderer.world().write_component::<component::Camera>();
            let camera = camera_comp.get_mut(entity).unwrap();

            let ms = Self::MOVEMENT_SPEED * delta_time as f32;

            let mut pos = camera.position();
            pos.y += vel_up_down as f32 * ms;

            camera.set_position(pos);
            camera.move2(vel_front_back as f32 * ms, vel_left_right as f32 * ms);

            let mut rotation = camera.rotation();
            let cursor_offset = (
                self.cursor_rel.0 as f32 * Self::MOUSE_SENSITIVITY,
                self.cursor_rel.1 as f32 * Self::MOUSE_SENSITIVITY,
            );

            rotation.x = na::clamp(
                rotation.x - cursor_offset.1,
                -std::f32::consts::FRAC_PI_2,
                std::f32::consts::FRAC_PI_2,
            );
            rotation.y += cursor_offset.0;

            camera.set_rotation(rotation);

            self.cursor_rel = (0, 0);
        }
    }
}

pub fn new(renderer: &Arc<Mutex<Renderer>>) -> Program {
    Program {
        renderer: Arc::clone(renderer),
        pressed_keys: Default::default(),
        cursor_rel: (0, 0),
    }
}
