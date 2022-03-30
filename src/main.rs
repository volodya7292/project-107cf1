mod utils;
#[macro_use]
mod game;
mod material_pipelines;
#[cfg(test)]
mod tests;

use crate::game::Game;
use engine::renderer::{Renderer, TextureQuality, TranslucencyMaxDepth};
use engine::resource_file::ResourceFile;
use engine::utils::noise::ParamNoise;
use engine::utils::thread_pool::SafeThreadPool;
use engine::{renderer, vertex_impl, Engine};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec2, Vec2, Vec3};
use noise::Seedable;
use simple_logger::SimpleLogger;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use winit::dpi::PhysicalPosition;
use winit::event::{VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Fullscreen, WindowBuilder};

#[derive(Copy, Clone, Default)]
pub struct BasicVertex {
    position: Vec3,
    tex_coord: Vec2,
}

vertex_impl!(BasicVertex, position, tex_coord);

const PROGRAM_NAME: &str = "project-107cf1";
const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);

fn noise_test() {
    let mut buf = vec![0_u8; 1024 * 1024 * 3];

    let n = noise::SuperSimplex::new().set_seed(0);

    let process = |p: DVec2| -> f64 {
        let v = n.sample(p, 5.0, 1.0, 0.5);

        let d = (glm::distance(&DVec2::new(0.5, 0.5), &p) * 2.0).min(1.0);
        let s = glm::smoothstep(0.0, 1.0, 4.0 * (1.0 - d));

        // y = 0.4 - (0.1 - (x - 0.4)) / 0.1

        v * s
    };

    for x in 0..1024 {
        for y in 0..1024 {
            let i = (y * 1024 + x) * 3;
            let x = x as f64 / 1024.0;
            let y = y as f64 / 1024.0;

            let v = process(DVec2::new(x, y));

            buf[i] = (v * 255.0) as u8;
            buf[i + 1] = buf[i];
            buf[i + 2] = buf[i];
        }
    }

    image::save_buffer("D:/noise_test.png", &buf, 1024, 1024, image::ColorType::Rgb8).unwrap();
    std::process::exit(0);
}

fn main() {
    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(PROGRAM_NAME)
        .with_inner_size(winit::dpi::PhysicalSize::new(
            DEF_WINDOW_SIZE.0,
            DEF_WINDOW_SIZE.1,
        ))
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();

    // Center the window
    {
        let win_size = window.outer_size();
        let mon_size = window.current_monitor().unwrap().size();

        window.set_outer_position(PhysicalPosition {
            x: (mon_size.width as i32 - win_size.width as i32) / 2,
            y: (mon_size.height as i32 - win_size.height as i32) / 2,
        });
    }

    let vke = vk_wrapper::Entry::new().unwrap();
    let instance = vke.create_instance(PROGRAM_NAME, &window).unwrap();

    let surface = instance.create_surface(&window).unwrap();

    let adapters = instance.enumerate_adapters(Some(&surface)).unwrap();
    let adapter = adapters.first().unwrap();
    let device = adapter.create_device().unwrap();

    let mut game = Box::new(Game::init());
    let mut engine = Engine::init(&surface, &device, 4, game);

    let mut cursor_grab = true;
    let mut start_t = Instant::now();

    window.set_cursor_grab(cursor_grab).unwrap();
    window.set_cursor_visible(!cursor_grab);

    event_loop.run_return(|event, _, control_flow| {
        use winit::event::ElementState;
        use winit::event::Event;

        *control_flow = ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                start_t = Instant::now();
            }
            Event::WindowEvent { window_id, ref event } => match event {
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input,
                    is_synthetic: _,
                } => {
                    if let Some(keycode) = input.virtual_keycode {
                        match keycode {
                            VirtualKeyCode::Escape => {
                                *control_flow = ControlFlow::Exit;
                            }
                            VirtualKeyCode::F11 => {
                                if input.state == ElementState::Released {
                                    if let Some(_) = window.fullscreen() {
                                        window.set_fullscreen(None);
                                    } else {
                                        let mode = utils::find_largest_video_mode(
                                            &window.current_monitor().unwrap(),
                                        );
                                        window.set_fullscreen(Some(Fullscreen::Exclusive(mode)))
                                    }
                                }
                            }
                            VirtualKeyCode::T => {
                                if input.state == ElementState::Released {
                                    cursor_grab = !cursor_grab;
                                    window.set_cursor_grab(cursor_grab).unwrap();
                                    window.set_cursor_visible(!cursor_grab);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::CloseRequested if window_id == window.id() => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            _ => {}
        }

        engine.on_winit_event(event);
    });
}
