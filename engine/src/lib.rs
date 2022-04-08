pub mod ecs;
pub mod input;
mod platform;
pub mod renderer;
pub mod resource_file;
#[cfg(test)]
mod tests;
pub mod utils;

use crate::input::{Keyboard, Mouse};
use crate::platform::current_refresh_rate;
use crate::renderer::{FPSLimit, Renderer, RendererTimings};
use crate::utils::thread_pool::{SafeThreadPool, ThreadPoolPriority};
use crate::utils::{HashSet, MO_RELAXED};
use lazy_static::lazy_static;
use rayon::ThreadPool;
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use vk_wrapper as vkw;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

lazy_static! {
    static ref INITIALIZED: AtomicBool = AtomicBool::new(false);
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
    render_tp: SafeThreadPool,
    update_tp: SafeThreadPool,
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
    fn on_update(
        &mut self,
        delta_time: f64,
        renderer: &mut Renderer,
        input: &mut Input,
        background_thread_pool: &ThreadPool,
    );
    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        main_window: &Window,
        control_flow: &mut ControlFlow,
    );
}

impl Engine {
    pub fn init(program_name: &str, max_texture_count: u32, mut app: Box<dyn Application + Send>) -> Engine {
        if INITIALIZED.swap(true, MO_RELAXED) {
            panic!("Engine has already been initialized!");
        }

        let n_threads = thread::available_parallelism().unwrap().get().max(2);
        let n_render_threads = (n_threads / 2).max(4);
        let n_update_threads = n_threads - n_render_threads;

        // Note: use safe thread pools to account for proper destruction of Vulkan objects.
        let render_thread_pool = SafeThreadPool::new(n_render_threads, ThreadPoolPriority::Realtime).unwrap();
        let update_thread_pool =
            SafeThreadPool::new(n_update_threads, ThreadPoolPriority::Background).unwrap();

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

        app.on_engine_initialized(&mut renderer);

        let curr_mode_refresh_rate = current_refresh_rate(&main_window);

        Engine {
            renderer,
            render_tp: render_thread_pool,
            update_tp: update_thread_pool,
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
                            self.renderer.on_resize((size.width, size.height));
                        }
                    }
                    _ => {}
                },
                Event::MainEventsCleared => {
                    // let t0 = Instant::now();

                    // println!("HIDDEN {}", (t0 - self.frame_end_time).as_secs_f64());

                    self.render_tp.install(|| {
                        let t0 = Instant::now();
                        self.app.on_update(
                            self.delta_time,
                            &mut self.renderer,
                            &mut self.input,
                            &self.update_tp,
                        );
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
                        utils::high_precision_sleep(
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

            self.app.on_event(event, &self.main_window, control_flow);
        });
    }
}
