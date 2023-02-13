pub mod ecs;
pub mod event;
pub mod execution;
pub mod input;
pub mod module;
mod platform;
#[cfg(test)]
mod tests;
pub mod utils;

use crate::ecs::dirty_components::DirtyComponents;
use crate::ecs::SceneAccess;
use crate::event::{WEvent, WSIEvent, WWindowEvent};
use crate::input::{Keyboard, Mouse};
use crate::module::{EngineModule, ModuleManager};
use crate::utils::wsi::WSISize;
use common::any::AsAny;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRef, OwnedRefMut};
use common::types::HashMap;
use common::MO_RELAXED;
use entity_data::EntityStorage;
use lazy_static::lazy_static;
pub use platform::Platform;
use std::any::{Any, TypeId};
use std::cell::{Ref, RefCell};
use std::fmt::{Display, Formatter};
use std::rc::Rc;
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
    main_window: Lrc<Window>,
    input: Input,
    curr_mode_refresh_rate: u32,
    curr_statistics: EngineStatistics,
    storage: Lrc<EntityStorage>,
    dirty_comps: Lrc<DirtyComponents>,
    object_count: Lrc<usize>,
    module_manager: Lrc<ModuleManager>,
    app: Lrc<dyn Application>,
}

pub struct EngineContext {
    storage: Lrc<EntityStorage>,
    dirty_comps: Lrc<DirtyComponents>,
    object_count: Lrc<usize>,
    module_manager: Lrc<ModuleManager>,
    app: Lrc<dyn Application>,
    window: Lrc<Window>,
}

impl EngineContext {
    pub fn window(&self) -> Ref<Window> {
        self.window.borrow()
    }

    pub fn scene(&self) -> SceneAccess {
        SceneAccess::new(self)
    }

    pub fn register_module<M: EngineModule>(&self, module: M) {
        self.module_manager.borrow_mut().register_module(module);
    }

    pub fn module_mut<M: EngineModule>(&self) -> OwnedRefMut<dyn EngineModule, M> {
        self.module_manager.borrow().module_mut()
    }

    pub fn app<T: Application>(&self) -> OwnedRef<dyn Application, T> {
        OwnedRef::map(self.app.borrow_owned(), |v| {
            v.as_any().downcast_ref::<T>().unwrap()
        })
    }
}

pub trait Application: AsAny + Send {
    fn on_engine_start(&mut self, event_loop: &EventLoop<()>) -> Window;
    fn initialize_engine(&mut self, _: &EngineContext);
    fn on_update(&mut self, delta_time: f64, ctx: &EngineContext, input: &mut Input);
    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        main_window: &Window,
        control_flow: &mut ControlFlow,
        ctx: &EngineContext,
    );
}

impl Engine {
    pub fn init(app: impl Application) -> Engine {
        if ENGINE_INITIALIZED.swap(true, MO_RELAXED) {
            panic!("Engine has already been initialized!");
        }

        let app = Rc::new(RefCell::new(app));

        let event_loop = EventLoop::new();
        let main_window = app.borrow_mut().on_engine_start(&event_loop);

        let curr_monitor = main_window.current_monitor().unwrap();
        let curr_mode_refresh_rate = curr_monitor.refresh_rate_millihertz().unwrap() / 1000;

        let mut engine = Engine {
            last_frame_end_time: Instant::now(),
            delta_time: 1.0,
            event_loop: Some(event_loop),
            main_window: Lrc::wrap(main_window),
            input: Input {
                keyboard: Keyboard::new(),
                mouse: Mouse::new(),
            },
            curr_mode_refresh_rate,
            curr_statistics: Default::default(),
            storage: Default::default(),
            dirty_comps: Rc::new(RefCell::new(Default::default())),
            object_count: Rc::new(RefCell::new(0)),
            module_manager: Default::default(),
            app: app.clone(),
        };

        app.borrow_mut().initialize_engine(&engine.context());

        engine
    }

    fn context(&self) -> EngineContext {
        EngineContext {
            storage: self.storage.clone(),
            dirty_comps: self.dirty_comps.clone(),
            object_count: self.object_count.clone(),
            module_manager: self.module_manager.clone(),
            app: self.app.clone(),
            window: self.main_window.clone(),
        }
    }

    pub fn run(mut self) {
        let mut event_loop = self.event_loop.take().unwrap();

        event_loop.run_return(move |event, _, control_flow| {
            use winit::event::ElementState;

            *control_flow = ControlFlow::Poll;

            match &event {
                WEvent::WindowEvent {
                    window_id: _window_id,
                    event,
                } => match event {
                    WWindowEvent::KeyboardInput { input, .. } => {
                        if let Some(keycode) = input.virtual_keycode {
                            if input.state == ElementState::Pressed {
                                self.input.keyboard.pressed_keys.insert(keycode);
                            } else {
                                self.input.keyboard.pressed_keys.remove(&keycode);
                            }
                        }
                    }
                    WWindowEvent::MouseInput { state, button, .. } => {
                        if *state == ElementState::Pressed {
                            self.input.mouse.pressed_buttons.insert(*button);
                        } else {
                            self.input.mouse.pressed_buttons.remove(button);
                        }
                    }
                    _ => {}
                },
                WEvent::MainEventsCleared => {
                    // let t0 = Instant::now();

                    // println!("HIDDEN {}", (t0 - self.frame_end_time).as_secs_f64());

                    let mut app = self.app.borrow_mut();
                    let module_manger = &*self.module_manager.borrow();
                    let delta_time = self.delta_time;
                    let ctx = self.context();

                    let t0 = Instant::now();

                    app.on_update(delta_time, &ctx, &mut self.input);
                    module_manger.on_update(&ctx);

                    let t1 = Instant::now();
                    self.curr_statistics.update_time = (t1 - t0).as_secs_f64();

                    let end_t = Instant::now();
                    self.delta_time = (end_t - self.last_frame_end_time).as_secs_f64().min(1.0);
                    self.last_frame_end_time = end_t;

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
                _ => {}
            }

            if let Some(event) = WSIEvent::from_winit(&event, &*self.main_window.borrow()) {
                self.module_manager.borrow().on_wsi_event(
                    &*self.main_window.borrow(),
                    &event,
                    &self.context(),
                );
            }

            self.app
                .borrow_mut()
                .on_event(event, &*self.main_window.borrow(), control_flow, &self.context());
        });
    }
}
