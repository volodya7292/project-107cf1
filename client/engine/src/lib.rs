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
use common::any::AsAny;
use common::lrc::{OwnedRef, OwnedRefMut};
use common::MO_RELAXED;
use entity_data::EntityStorage;
use lazy_static::lazy_static;
pub use platform::Platform;
use std::cell::{Ref, RefCell};
use std::fmt::{Display, Formatter};
use std::sync::atomic::AtomicBool;
use std::time::Instant;
pub use vk_wrapper as vkw;
pub use winit;
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
    last_frame_end_time: Instant,
    delta_time: f64,
    event_loop: Option<EventLoop<()>>,
    main_window: RefCell<Window>,
    curr_mode_refresh_rate: u32,
    curr_statistics: EngineStatistics,
    module_manager: RefCell<ModuleManager>,
    app: RefCell<Box<dyn Application>>,
}

pub struct EngineContext<'a> {
    module_manager: &'a RefCell<ModuleManager>,
    app: &'a RefCell<Box<dyn Application>>,
    window: &'a RefCell<Window>,
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

    pub fn app<T: Application>(&self) -> Ref<T> {
        Ref::map(self.app.borrow(), |app| app.as_any().downcast_ref::<T>().unwrap())
    }
}

// TODO: refactor Application into a module

pub trait Application: AsAny + Send {
    fn on_engine_start(&mut self, event_loop: &EventLoop<()>) -> Window;
    fn initialize_engine(&mut self, _: &EngineContext);
    fn on_update(&mut self, delta_time: f64, ctx: &EngineContext);
    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        main_window: &Window,
        control_flow: &mut ControlFlow,
        ctx: &EngineContext,
    );
}

impl Engine {
    pub fn init(mut app: impl Application) -> Self {
        if ENGINE_INITIALIZED.swap(true, MO_RELAXED) {
            panic!("Engine has already been initialized!");
        }

        let event_loop = EventLoop::new();
        let main_window = app.on_engine_start(&event_loop);

        let curr_monitor = main_window.current_monitor().unwrap();
        let curr_mode_refresh_rate = curr_monitor.refresh_rate_millihertz().unwrap() / 1000;

        let engine = Self {
            last_frame_end_time: Instant::now(),
            delta_time: 1.0,
            event_loop: Some(event_loop),
            main_window: RefCell::new(main_window),
            curr_mode_refresh_rate,
            curr_statistics: Default::default(),
            module_manager: Default::default(),
            app: RefCell::new(Box::new(app)),
        };

        engine.app.borrow_mut().initialize_engine(&engine.context());

        engine
    }

    fn context(&self) -> EngineContext {
        EngineContext {
            module_manager: &self.module_manager,
            app: &self.app,
            window: &self.main_window,
        }
    }

    pub fn run(mut self) {
        let mut event_loop = self.event_loop.take().unwrap();

        event_loop.run_return(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;

            match &event {
                WEvent::MainEventsCleared => {
                    // let t0 = Instant::now();

                    // println!("HIDDEN {}", (t0 - self.frame_end_time).as_secs_f64());

                    let mut app = self.app.borrow_mut();
                    let module_manger = &*self.module_manager.borrow();
                    let ctx = self.context();

                    let t0 = Instant::now();

                    app.on_update(self.delta_time, &ctx);
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
                _ => {
                    if let Some(event) = WSIEvent::from_winit(&event, &*self.main_window.borrow()) {
                        self.module_manager.borrow().on_wsi_event(
                            &*self.main_window.borrow(),
                            &event,
                            &self.context(),
                        );
                    }

                    self.app.borrow_mut().on_event(
                        event,
                        &*self.main_window.borrow(),
                        control_flow,
                        &self.context(),
                    );
                }
            }
        });
    }
}
