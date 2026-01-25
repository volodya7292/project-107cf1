#[macro_use]
pub mod ecs;
pub mod event;
pub mod execution;
pub mod gltf;
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
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
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
    initialized: bool,
    do_stop: AtomicBool,
    last_frame_end_time: Instant,
    delta_time: f64,
    event_loop: Option<EventLoop<()>>,
    on_init: Box<dyn Fn(&EngineContext)>,
    create_window_fn: Box<dyn Fn(&ActiveEventLoop) -> Window>,
    main_window: Option<RefCell<Window>>,
    curr_sizing_info: Option<WSizingInfo>,
    curr_mode_refresh_rate: u32,
    curr_statistics: EngineStatistics,
    module_manager: RefCell<ModuleManager>,
    callbacks: RefCell<Vec<Box<dyn FnOnce(&EngineContext, f64)>>>,
}

#[derive(Copy, Clone)]
pub struct EngineContext<'a> {
    do_stop: &'a AtomicBool,
    module_manager: &'a RefCell<ModuleManager>,
    window: &'a RefCell<Window>,
    callbacks: &'a RefCell<Vec<Box<dyn FnOnce(&EngineContext, f64)>>>,
}

impl EngineContext<'_> {
    pub fn window(&self) -> Ref<'_, Window> {
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

    pub fn module_last_update_time<M: EngineModule>(&self) -> f64 {
        self.module_manager.borrow().module_last_update_time::<M>()
    }

    pub fn dispatch_callback<F: FnOnce(&EngineContext, f64) + 'static>(&self, callback: F) {
        self.callbacks.borrow_mut().push(Box::new(callback));
    }

    pub fn request_stop(&self) {
        self.do_stop.store(true, MO_SEQCST);
    }
}

impl Engine {
    pub fn init<F: Fn(&ActiveEventLoop) -> Window + 'static, IF: Fn(&EngineContext) + 'static>(
        window_init: F,
        on_init: IF,
    ) -> Self {
        if ENGINE_INITIALIZED.swap(true, MO_RELAXED) {
            panic!("Engine has already been initialized!");
        }

        let event_loop = EventLoop::new().unwrap();

        let engine = Self {
            initialized: false,
            do_stop: Default::default(),
            last_frame_end_time: Instant::now(),
            delta_time: 1.0,
            event_loop: Some(event_loop),
            on_init: Box::new(on_init),
            create_window_fn: Box::new(window_init),
            main_window: None,
            curr_sizing_info: None,
            curr_mode_refresh_rate: 60,
            curr_statistics: Default::default(),
            module_manager: Default::default(),
            callbacks: RefCell::new(vec![]),
        };

        engine
    }

    pub fn context(&self) -> EngineContext<'_> {
        EngineContext {
            do_stop: &self.do_stop,
            module_manager: &self.module_manager,
            window: self.main_window.as_ref().unwrap(),
            callbacks: &self.callbacks,
        }
    }

    pub fn run(mut self) {
        let event_loop = self.event_loop.take().unwrap();
        event_loop.run_app(&mut self).unwrap();
    }
}

impl ApplicationHandler for Engine {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {}

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        event_loop.set_control_flow(ControlFlow::Poll);

        if self.do_stop.load(MO_SEQCST) {
            event_loop.exit();
            return;
        }

        // we don't use StartCause::Init here as it stalls the process if initialization panics
        if cause == winit::event::StartCause::Poll && self.initialized == false {
            let main_window = (self.create_window_fn)(event_loop);
            let sizing_info = WSizingInfo::get(&main_window);

            let curr_monitor = main_window.current_monitor().unwrap();
            let _curr_mode_refresh_rate = curr_monitor.refresh_rate_millihertz().unwrap() / 1000;

            self.main_window = Some(RefCell::new(main_window));
            self.curr_sizing_info = Some(sizing_info);

            self.module_manager.borrow().on_start(&self.context());
            (self.on_init)(&self.context());
            self.initialized = true;
        }

        if cause == winit::event::StartCause::Poll {
            // let t0 = Instant::now();

            // println!("HIDDEN {}", (t0 - self.frame_end_time).as_secs_f64());

            let module_manger = &*self.module_manager.borrow();
            let ctx = self.context();

            let t0 = Instant::now();

            for (module_id, _module) in module_manger.modules() {
                module_manger.update_module(module_id, self.delta_time, &ctx);

                // Call dispatched callbacks as soon as possible
                let callbacks: Vec<_> = self.callbacks.borrow_mut().drain(..).collect();
                for callback in callbacks {
                    callback(&ctx, self.delta_time);
                }
            }

            let t1 = Instant::now();
            self.curr_statistics.update_time = (t1 - t0).as_secs_f64();

            let curr_end_t = Instant::now();
            self.delta_time = (curr_end_t - self.last_frame_end_time).as_secs_f64().min(1.0);
            self.last_frame_end_time = curr_end_t;

            // let t1 = Instant::now();
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        let event = WEvent::DeviceEvent { device_id, event };
        let main_window = self.main_window.as_ref().unwrap().borrow();

        if let Some(event) =
            WSIEvent::from_winit(&event, &main_window, self.curr_sizing_info.as_ref().unwrap())
        {
            self.module_manager
                .borrow()
                .on_wsi_event(&main_window, &event, &self.context());
        }
    }

    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match &event {
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                let main_window = self.main_window.as_ref().unwrap().borrow();
                self.curr_sizing_info = Some(WSizingInfo::get(&main_window));
            }
            WindowEvent::CloseRequested => {
                self.do_stop.store(true, MO_SEQCST);
            }
            _ => {} // },
                    // _ => {}
        }

        let event = WEvent::WindowEvent { window_id, event };
        let main_window = self.main_window.as_ref().unwrap().borrow();

        if let Some(event) =
            WSIEvent::from_winit(&event, &main_window, self.curr_sizing_info.as_ref().unwrap())
        {
            self.module_manager
                .borrow()
                .on_wsi_event(&main_window, &event, &self.context());
        }
    }
}
