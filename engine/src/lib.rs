pub mod ecs;
pub mod keyboard;
pub mod renderer;
pub mod resource_file;
#[cfg(test)]
mod tests;
pub mod utils;

use crate::keyboard::Keyboard;
use crate::renderer::Renderer;
use crate::utils::thread_pool::SafeThreadPool;
use crate::utils::HashSet;
use rayon::ThreadPool;
use std::sync::Arc;
use std::thread;
use std::time::Instant;
use vk_wrapper as vkw;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

pub struct Input {
    keyboard: Keyboard,
}

impl Input {
    pub fn keyboard(&self) -> &Keyboard {
        &self.keyboard
    }
}

pub struct Engine {
    renderer: Renderer,
    render_tp: SafeThreadPool,
    update_tp: SafeThreadPool,
    frame_start_time: Instant,
    delta_time: f64,
    event_loop: EventLoop<()>,
    main_window: Window,
    input: Input,
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
        let n_threads = thread::available_parallelism().unwrap().get().max(2);
        let n_render_threads = (n_threads / 2).max(4);
        let n_update_threads = n_threads - n_render_threads;

        // Note: use safe thread pools to account for proper destruction of Vulkan objects.
        let render_thread_pool = SafeThreadPool::new(n_render_threads).unwrap();
        let update_thread_pool = SafeThreadPool::new(n_update_threads).unwrap();

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

        Engine {
            renderer,
            render_tp: render_thread_pool,
            update_tp: update_thread_pool,
            frame_start_time: Instant::now(),
            delta_time: 1.0,
            event_loop,
            main_window,
            input: Input {
                keyboard: Keyboard::new(),
            },
            app,
        }
    }

    pub fn run(&mut self) {
        self.event_loop.run_return(|event, _, control_flow| {
            use winit::event::ElementState;
            use winit::event::Event;

            *control_flow = ControlFlow::Poll;

            match &event {
                Event::NewEvents(_) => {
                    self.frame_start_time = Instant::now();
                }
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

                        if let Some(keycode) = input.virtual_keycode {
                            match keycode {
                                _ => {}
                            }
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
                    self.render_tp.install(|| {
                        self.app.on_update(
                            self.delta_time,
                            &mut self.renderer,
                            &mut self.input,
                            &self.update_tp,
                        );
                        self.renderer.on_draw();
                    });
                }
                Event::RedrawEventsCleared => {
                    let end_dt = Instant::now();
                    self.delta_time = (end_dt - self.frame_start_time).as_secs_f64();
                }
                _ => {}
            }

            self.app.on_event(event, &self.main_window, control_flow);
        });
    }
}
