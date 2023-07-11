#[macro_use]
pub mod ecs;
pub mod event;
pub mod execution;
pub mod module;
mod platform;
#[cfg(test)]
mod tests;
pub mod utils;

use crate::event::{WEvent, WSIEvent};
use crate::module::{EngineModule, ModuleManager};
use crate::utils::wsi::vec2::WSizingInfo;
use common::lrc::{OwnedRef, OwnedRefMut};
use common::{MO_RELAXED, MO_SEQCST};
use lazy_static::lazy_static;
pub use platform::Platform;
use std::cell::{Ref, RefCell};
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
pub use vk_wrapper as vkw;
pub use winit;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::Window;

lazy_static! {
    static ref ENGINE_INITIALIZED: AtomicBool = AtomicBool::new(false);
}

#[derive(Default, Copy, Clone)]
pub struct EngineStatistics {
    update_time: f64,
}

impl EngineStatistics {
    pub fn total(&self) -> f64 {
        self.update_time
    }
}

impl Display for EngineStatistics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("Engine Statistics: upd {:.5}", self.update_time))
    }
}

pub struct Engine {
    do_stop: AtomicBool,
    last_frame_end_time: Instant,
    delta_time: f64,
    event_loop: Option<EventLoop<()>>,
    main_window: RefCell<Window>,
    curr_sizing_info: WSizingInfo,
    curr_mode_refresh_rate: u32,
    curr_statistics: EngineStatistics,
    module_manager: RefCell<ModuleManager>,
    callbacks: RefCell<Vec<Box<dyn FnOnce(&EngineContext, f64)>>>,
}

pub struct EngineContext<'a> {
    do_stop: &'a AtomicBool,
    module_manager: &'a RefCell<ModuleManager>,
    window: &'a RefCell<Window>,
    callbacks: &'a RefCell<Vec<Box<dyn FnOnce(&EngineContext, f64)>>>,
}

impl EngineContext<'_> {
    pub fn window(&self) -> Ref<Window> {
        self.window.borrow()
    }

    pub fn register_module<M: EngineModule>(&self, module: M) {
        self.module_manager.borrow_mut().register_module(module);
    }

    pub fn module<M: EngineModule>(&self) -> OwnedRef<dyn EngineModule, M> {
        self.module_manager.borrow().module()
    }

    pub fn module_mut<M: EngineModule>(&self) -> OwnedRefMut<dyn EngineModule, M> {
        self.module_manager.borrow().module_mut()
    }

    pub fn dispatch_callback<F: FnOnce(&EngineContext, f64) + 'static>(&self, callback: F) {
        self.callbacks.borrow_mut().push(Box::new(callback));
    }

    pub fn request_stop(&self) {
        self.do_stop.store(true, MO_SEQCST);
    }
}

impl Engine {
    pub fn init<F: FnOnce(&EventLoop<()>) -> Window>(window_init: F) -> Self {
        if ENGINE_INITIALIZED.swap(true, MO_RELAXED) {
            panic!("Engine has already been initialized!");
        }

        let event_loop = EventLoop::new();
        let main_window = window_init(&event_loop);
        let sizing_info = WSizingInfo::get(&main_window);

        let curr_monitor = main_window.current_monitor().unwrap();
        let curr_mode_refresh_rate = curr_monitor.refresh_rate_millihertz().unwrap() / 1000;

        let engine = Self {
            do_stop: Default::default(),
            last_frame_end_time: Instant::now(),
            delta_time: 1.0,
            event_loop: Some(event_loop),
            main_window: RefCell::new(main_window),
            curr_sizing_info: sizing_info,
            curr_mode_refresh_rate,
            curr_statistics: Default::default(),
            module_manager: Default::default(),
            callbacks: RefCell::new(vec![]),
        };

        engine
    }

    pub fn context(&self) -> EngineContext {
        EngineContext {
            do_stop: &self.do_stop,
            module_manager: &self.module_manager,
            window: &self.main_window,
            callbacks: &self.callbacks,
        }
    }

    pub fn run(mut self) {
        self.module_manager.borrow().on_start(&self.context());

        let mut event_loop = self.event_loop.take().unwrap();

        event_loop.run_return(move |event, _, control_flow| {
            *control_flow = if self.do_stop.load(MO_SEQCST) {
                ControlFlow::ExitWithCode(0)
            } else {
                ControlFlow::Poll
            };

            match &event {
                WEvent::MainEventsCleared => {
                    // let t0 = Instant::now();

                    // println!("HIDDEN {}", (t0 - self.frame_end_time).as_secs_f64());

                    let module_manger = &*self.module_manager.borrow();
                    let ctx = self.context();

                    let t0 = Instant::now();

                    // Call dispatched callbacks here so any module can be borrowed in the handler
                    let callbacks: Vec<_> = self.callbacks.borrow_mut().drain(..).collect();
                    for callback in callbacks {
                        callback(&ctx, self.delta_time);
                    }
                    module_manger.on_update(self.delta_time, &ctx);

                    let t1 = Instant::now();
                    self.curr_statistics.update_time = (t1 - t0).as_secs_f64();

                    let curr_end_t = Instant::now();
                    self.delta_time = (curr_end_t - self.last_frame_end_time).as_secs_f64().min(1.0);
                    self.last_frame_end_time = curr_end_t;

                    // let t1 = Instant::now();
                    // self.curr_statistics.total = (t1 - t0).as_secs_f64();
                }
                // WEvent::RedrawEventsCleared => {
                //     let expected_dt;
                //
                //     if let FPSLimit::Limit(limit) = self.renderer.settings().fps_limit {
                //         expected_dt = 1.0 / (limit as f64);
                //         let to_wait = (expected_dt - self.delta_time).max(0.0);
                //         base::utils::high_precision_sleep(
                //             Duration::from_secs_f64(to_wait),
                //             Duration::from_micros(50),
                //         );
                //         self.delta_time += to_wait;
                //     } else {
                //         // expected_dt = 1.0 / self.curr_mode_refresh_rate as f64;
                //     }
                //
                //     // if self.delta_time >= (1.0 / self.curr_mode_refresh_rate as f64) {
                //     let total_frame_time = self.curr_statistics.total();
                //     if total_frame_time >= 0.017 {
                //         println!(
                //             "dt {:.5}| total {:.5} | {}",
                //             self.delta_time, total_frame_time, self.curr_statistics
                //         );
                //     }
                //
                //     // println!("dt {}", self.delta_time);
                //     self.last_frame_end_time = Instant::now();
                //     self.frame_start_time = self.last_frame_end_time;
                // }
                WEvent::WindowEvent { window_id: _, event } => match event {
                    WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                        self.curr_sizing_info = WSizingInfo::get(&self.main_window.borrow());
                    }
                    WindowEvent::CloseRequested => {
                        self.do_stop.store(true, MO_SEQCST);
                    }
                    _ => {}
                },
                _ => {}
            }

            if let Some(event) =
                WSIEvent::from_winit(&event, &*self.main_window.borrow(), &self.curr_sizing_info)
            {
                self.module_manager.borrow().on_wsi_event(
                    &*self.main_window.borrow(),
                    &event,
                    &self.context(),
                );
            }
        });
    }
}
