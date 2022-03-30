pub mod ecs;
pub mod physics;
pub mod renderer;
pub mod resource_file;
pub mod utils;

use crate::ecs::scene::Scene;
use crate::renderer::Renderer;
use crate::utils::thread_pool::SafeThreadPool;
use parking_lot::Mutex;
use rayon::ThreadPool;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc};
use std::thread;
use std::time::{Duration, Instant};
use vk_wrapper as vkw;
use vk_wrapper::Surface;
use winit::event::{Event, WindowEvent};

pub struct Engine {
    renderer: Renderer,
    render_tp: SafeThreadPool,
    update_tp: SafeThreadPool,
    frame_start_time: Instant,
    delta_time: f64,
    app: Box<dyn Application>,
}

pub trait Application {
    fn on_start(&mut self, renderer: &mut Renderer);
    fn on_update(&mut self, delta_time: f32, renderer: &mut Renderer, background_thread_pool: &ThreadPool);
    fn on_event(&mut self, event: winit::event::Event<()>);
}

impl Engine {
    pub fn init(
        surface: &Arc<Surface>,
        device: &Arc<vkw::Device>,
        max_texture_count: u32,
        mut app: Box<dyn Application>,
    ) -> Engine {
        let n_threads = thread::available_parallelism().unwrap().get().max(2);
        let n_render_threads = (n_threads / 2).max(4);
        let n_update_threads = n_threads - n_render_threads;

        // Note: use safe thread pools to account for proper destruction of Vulkan objects.
        let render_thread_pool = SafeThreadPool::new(n_render_threads).unwrap();
        let update_thread_pool = SafeThreadPool::new(n_update_threads).unwrap();

        let mut renderer = Renderer::new(
            surface,
            (1280, 720),
            Default::default(),
            device,
            max_texture_count,
        );

        app.on_start(&mut renderer);

        Engine {
            renderer,
            render_tp: render_thread_pool,
            update_tp: update_thread_pool,
            frame_start_time: Instant::now(),
            delta_time: 1.0,
            app,
        }
    }

    pub fn on_winit_event(&mut self, event: winit::event::Event<()>) {
        match &event {
            Event::NewEvents(_) => {
                self.frame_start_time = Instant::now();
            }
            Event::WindowEvent {
                window_id: _window_id,
                event,
            } => match event {
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        self.renderer.on_resize((size.width, size.height));
                    }
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                self.app
                    .on_update(self.delta_time as f32, &mut self.renderer, &self.update_tp);

                // TODO: handle physics here

                self.render_tp.install(|| self.renderer.on_draw());
            }
            Event::RedrawEventsCleared => {
                let end_dt = Instant::now();
                self.delta_time = (end_dt - self.frame_start_time).as_secs_f64();
            }
            _ => {}
        }

        self.app.on_event(event);
    }
}
