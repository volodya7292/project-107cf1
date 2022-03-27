mod utils;

#[macro_use]
pub(crate) mod render_engine;
mod game;
mod resource_file;

#[cfg(test)]
mod tests;

use crate::render_engine::{component, TextureQuality, TranslucencyMaxDepth};
use crate::render_engine::{material_pipeline, material_pipelines};
use crate::resource_file::ResourceFile;
use crate::utils::noise::ParamNoise;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec2, Vec2, Vec3};
use noise::Seedable;
use std::path::Path;
use std::thread;
use std::time::Instant;
use utils::thread_pool::SafeThreadPool;
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
    simple_logger::SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();

    let n_threads = thread::available_parallelism().unwrap().get().max(2);
    let n_render_threads = (n_threads / 2).max(4);
    let n_update_threads = n_threads - n_render_threads;

    // Note: use safe thread pools to account for proper destruction of Vulkan objects.
    let render_thread_pool = SafeThreadPool::new(n_render_threads).unwrap();
    let update_thread_pool = SafeThreadPool::new(n_update_threads).unwrap();

    let mut resources = ResourceFile::open(Path::new("resources")).unwrap();

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

    let window_size = window.inner_size();

    let renderer_settings = render_engine::Settings {
        vsync: true,
        texture_quality: TextureQuality::STANDARD,
        translucency_max_depth: TranslucencyMaxDepth::LOW,
        textures_gen_mipmaps: true,
        textures_max_anisotropy: 1.0,
    };
    let renderer = render_engine::new(
        &surface,
        (window_size.width, window_size.height),
        renderer_settings,
        &device,
        &mut resources,
        4,
    );

    let mut program = game::new(&renderer, &resources);
    let mut cursor_grab = true;
    let mut start_t = Instant::now();
    let mut delta_time = 0.0;

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
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        renderer.lock().on_resize((size.width, size.height));
                    }
                }
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
            Event::MainEventsCleared => {
                program.on_update(delta_time, &update_thread_pool);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                render_thread_pool.install(|| renderer.lock().on_draw());
            }
            Event::RedrawEventsCleared => {
                let end_t = Instant::now();
                delta_time = (end_t - start_t).as_secs_f64();
            }
            _ => {}
        }

        program.on_event(event);
    });
}
