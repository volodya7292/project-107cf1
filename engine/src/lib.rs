pub mod ecs;
pub mod execution;
pub mod input;
mod platform;
pub mod renderer;
#[cfg(test)]
mod tests;
pub mod utils;

use crate::execution::realtime_queue;
use crate::input::{Keyboard, Mouse};
use crate::platform::{Platform, PlatformImpl};
use crate::renderer::module::text_renderer::TextRenderer;
use crate::renderer::module::ui_renderer::UIRenderer;
use crate::renderer::{FPSLimit, Renderer, RendererTimings};
use base::utils::{HashSet, MO_RELAXED};
use lazy_static::lazy_static;
use nalgebra_glm::Vec2;
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};
use vk_wrapper as vkw;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::monitor::VideoMode;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

lazy_static! {
    static ref ENGINE_INITIALIZED: AtomicBool = AtomicBool::new(false);
}

pub struct Input {
    keyboard: Keyboard,
    mouse: Mouse,
}

impl Input {
    pub fn keyboard(&self) -> &Keyboard {
        &self.keyboard
    }

    pub fn mouse(&self) -> &Mouse {
        &self.mouse
    }
}

#[derive(Default, Copy, Clone)]
pub struct EngineStatistics {
    update_time: f64,
    render_time: RendererTimings,
}

impl EngineStatistics {
    pub fn total(&self) -> f64 {
        self.update_time + self.render_time.update.total + self.render_time.render.total
    }
}

impl Display for EngineStatistics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "Engine Statistics: upd {:.5} | renderer: {:.5}",
            self.update_time, self.render_time
        ))
    }
}

pub struct Engine {
    renderer: Renderer,
    frame_start_time: Instant,
    frame_end_time: Instant,
    delta_time: f64,
    event_loop: EventLoop<()>,
    main_window: Window,
    input: Input,
    curr_mode_refresh_rate: u32,
    curr_statistics: EngineStatistics,
    app: Box<dyn Application + Send>,
}

pub trait Application {
    fn on_engine_start(&mut self, event_loop: &EventLoop<()>) -> Window;
    fn on_adapter_select(&mut self, adapters: &[Arc<vkw::Adapter>]) -> usize;
    fn on_engine_initialized(&mut self, renderer: &mut Renderer);
    fn on_update(&mut self, delta_time: f64, renderer: &mut Renderer, input: &mut Input);
    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        main_window: &Window,
        control_flow: &mut ControlFlow,
        renderer: &mut Renderer,
    );
}

/// Calculates scale factor relative to `video_mode`.
/// On macOS display size may be differ from current video mode size.
fn calc_real_scale_factor(curr_window: &Window) -> f64 {
    let curr_monitor = curr_window.current_monitor().unwrap();
    let mode = utils::find_best_video_mode(&curr_monitor);
    let display_width = curr_monitor.size();
    let scale_factor = curr_window.scale_factor();

    mode.size().width as f64 / display_width.width as f64 * scale_factor
}

impl Engine {
    pub fn init(program_name: &str, max_texture_count: u32, mut app: Box<dyn Application + Send>) -> Engine {
        if ENGINE_INITIALIZED.swap(true, MO_RELAXED) {
            panic!("Engine has already been initialized!");
        }

        let event_loop = EventLoop::new();
        let main_window = app.on_engine_start(&event_loop);

        let vke = vkw::Entry::new().unwrap();
        let instance = vke.create_instance(program_name, &main_window).unwrap();
        let surface = instance.create_surface(&main_window).unwrap();
        let adapters = instance.enumerate_adapters(Some(&surface)).unwrap();

        let adapter_index = app.on_adapter_select(&adapters);
        let adapter = &adapters[adapter_index];
        let device = adapter.create_device().unwrap();

        let mut renderer = Renderer::new(
            &surface,
            (1280, 720),
            Default::default(),
            &device,
            max_texture_count,
        );

        let text_renderer = TextRenderer::new(&mut renderer);
        let ui_renderer = UIRenderer::new(&mut renderer);

        renderer.register_module(text_renderer);
        renderer.register_module(ui_renderer);

        app.on_engine_initialized(&mut renderer);

        let curr_monitor = main_window.current_monitor().unwrap();
        let curr_mode_refresh_rate = curr_monitor.refresh_rate_millihertz().unwrap() / 1000;

        let dpi = Platform::get_monitor_dpi(&curr_monitor).unwrap();
        println!("{}", dpi);

        Engine {
            renderer,
            frame_start_time: Instant::now(),
            frame_end_time: Instant::now(),
            delta_time: 1.0,
            event_loop,
            main_window,
            input: Input {
                keyboard: Keyboard::new(),
                mouse: Mouse::new(),
            },
            curr_mode_refresh_rate,
            curr_statistics: Default::default(),
            app,
        }
    }

    pub fn run(mut self) {
        self.event_loop.run_return(move |event, _, control_flow| {
            use winit::event::ElementState;
            use winit::event::Event;

            *control_flow = ControlFlow::Poll;

            match &event {
                Event::WindowEvent {
                    window_id: _window_id,
                    event,
                } => match event {
                    WindowEvent::KeyboardInput {
                        device_id: _,
                        input,
                        is_synthetic: _,
                    } => {
                        if let Some(keycode) = input.virtual_keycode {
                            if input.state == ElementState::Pressed {
                                self.input.keyboard.pressed_keys.insert(keycode);
                            } else {
                                self.input.keyboard.pressed_keys.remove(&keycode);
                            }
                        }
                    }
                    WindowEvent::MouseInput {
                        device_id: _,
                        state,
                        button,
                        ..
                    } => {
                        if *state == ElementState::Pressed {
                            self.input.mouse.pressed_buttons.insert(*button);
                        } else {
                            self.input.mouse.pressed_buttons.remove(button);
                        }
                    }
                    WindowEvent::Resized(size) => {
                        if size.width != 0 && size.height != 0 {
                            let real_scale_factor = calc_real_scale_factor(&self.main_window);
                            self.renderer
                                .on_resize((size.width, size.height), real_scale_factor);
                        }
                    }
                    WindowEvent::ScaleFactorChanged {
                        scale_factor: _,
                        new_inner_size: size,
                    } => {
                        let real_scale_factor = calc_real_scale_factor(&self.main_window);
                        println!("scaling changed {}", real_scale_factor);
                        self.renderer
                            .on_resize((size.width, size.height), real_scale_factor);
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    // let t0 = Instant::now();

                    // println!("HIDDEN {}", (t0 - self.frame_end_time).as_secs_f64());

                    realtime_queue().install(|| {
                        let t0 = Instant::now();
                        self.app
                            .on_update(self.delta_time, &mut self.renderer, &mut self.input);
                        let t1 = Instant::now();
                        self.curr_statistics.update_time = (t1 - t0).as_secs_f64();

                        self.curr_statistics.render_time = self.renderer.on_draw();
                    });

                    // let t1 = Instant::now();
                    // self.curr_statistics.total = (t1 - t0).as_secs_f64();
                }
                Event::RedrawEventsCleared => {
                    let end_dt = Instant::now();
                    self.delta_time = (end_dt - self.frame_start_time).as_secs_f64().min(1.0);

                    let expected_dt;

                    if let FPSLimit::Limit(limit) = self.renderer.settings().fps_limit {
                        expected_dt = 1.0 / (limit as f64);
                        let to_wait = (expected_dt - self.delta_time).max(0.0);
                        base::utils::high_precision_sleep(
                            Duration::from_secs_f64(to_wait),
                            Duration::from_micros(50),
                        );
                        self.delta_time += to_wait;
                    } else {
                        // expected_dt = 1.0 / self.curr_mode_refresh_rate as f64;
                    }

                    // if self.delta_time >= (1.0 / self.curr_mode_refresh_rate as f64) {
                    let total_frame_time = self.curr_statistics.total();
                    if total_frame_time >= 0.017 {
                        println!(
                            "dt {:.5}| total {:.5} | {}",
                            self.delta_time, total_frame_time, self.curr_statistics
                        );
                    }

                    // println!("dt {}", self.delta_time);
                    self.frame_end_time = Instant::now();
                    self.frame_start_time = self.frame_end_time;
                }
                _ => {}
            }

            self.app
                .on_event(event, &self.main_window, control_flow, &mut self.renderer);
        });
    }
}
