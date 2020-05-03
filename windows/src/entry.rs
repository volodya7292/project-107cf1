use std::cell::Cell;

use log::error;

use crate::Window;

pub struct Entry {
    glfw_entry: glfw::Glfw,
}

fn glfw_error_callback(_: glfw::Error, description: String, _user_data: &Cell<usize>) {
    error!(target: "windows", "{}", description);
}

impl Entry {
    pub fn new() -> Result<Entry, glfw::InitError> {
        Ok(Entry {
            glfw_entry: glfw::init(Some(glfw::ErrorCallback {
                f: glfw_error_callback,
                data: Cell::new(0),
            }))?,
        })
    }

    pub fn get_required_vk_instance_extensions(&self) -> Option<Vec<String>> {
        self.glfw_entry.get_required_instance_extensions()
    }

    pub fn get_screen_size(&mut self) -> Option<(u32, u32)> {
        self.glfw_entry.with_primary_monitor_mut(|_, m| match m {
            Some(m) => {
                let vid_mode = m.get_video_mode().unwrap();
                Some((vid_mode.width, vid_mode.height))
            }
            _ => None,
        })
    }

    pub fn create_window(&mut self, width: u32, height: u32, title: &str) -> Option<Window> {
        self.glfw_entry
            .window_hint(glfw::WindowHint::ClientApi(glfw::ClientApiHint::NoApi));

        let (mut window, receiver) =
            self.glfw_entry
                .create_window(width, height, title, glfw::WindowMode::Windowed)?;

        let screen_size = self.get_screen_size()?;
        let pos = (
            (screen_size.0 as i32 - width as i32) / 2,
            (screen_size.1 as i32 - height as i32) / 2,
        );

        window.set_pos(pos.0, pos.1);
        window.set_framebuffer_size_polling(true);
        window.set_char_polling(true);
        window.set_key_polling(true);

        Some(Window {
            native: window,
            events: receiver,
            pos,
            size: (width, height),
        })
    }

    pub fn set_fullscreen(&mut self, window: &mut Window, fullscreen: bool) {
        self.glfw_entry.with_primary_monitor_mut(|_, m| match m {
            Some(m) if fullscreen => {
                let vid_mode = m.get_video_mode().unwrap();
                window.native.set_monitor(
                    glfw::WindowMode::FullScreen(m),
                    0,
                    0,
                    vid_mode.width,
                    vid_mode.height,
                    None,
                );
            }
            _ => {
                window.native.set_monitor(
                    glfw::WindowMode::Windowed,
                    window.pos.0,
                    window.pos.1,
                    window.size.0,
                    window.size.1,
                    None,
                );
            }
        });
    }

    pub fn main_loop(&mut self, windows: &[&Window]) {
        loop {
            self.glfw_entry.poll_events();

            let mut exit = true;
            for window in windows {
                if !window.native.should_close() {
                    exit = false;
                    break;
                }
            }
            if exit {
                break;
            }

            for window in windows {
                for (_, event) in glfw::flush_messages(&window.events) {
                    match event {
                        glfw::WindowEvent::Key(key, scancode, action, modifiers) => {
                            if action == glfw::Action::Press {
                                println!("Key: {:?}", key);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
