pub mod input;
pub mod main_renderer;
pub mod scene;
pub mod text_renderer;
pub mod ui;
pub mod ui_interaction_manager;

use crate::event::WSIEvent;
use crate::EngineContext;
use common::any::AsAny;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRef, OwnedRefMut};
use common::types::{HashMap, IndexMap};
use std::any::TypeId;
use std::cell::Cell;
use std::time::Instant;
use winit::window::Window;

pub trait EngineModule: AsAny {
    fn on_start(&mut self, _: &EngineContext) {}
    /// Main loop
    fn on_update(&mut self, _dt: f64, _: &EngineContext) {}
    fn on_wsi_event(&mut self, _: &Window, _: &WSIEvent, _: &EngineContext) {}
}

#[derive(Default)]
pub(crate) struct ModuleManager {
    modules: IndexMap<TypeId, Lrc<dyn EngineModule>>,
    last_module_update_times: HashMap<TypeId, Cell<f64>>,
}

impl ModuleManager {
    /// Registers a new `EngineModule`.
    /// Callbacks to all modules will be called in reversed registration order of the modules.
    pub fn register_module<M: EngineModule>(&mut self, module: M) {
        self.modules.insert(TypeId::of::<M>(), Lrc::wrap(module));
        self.last_module_update_times
            .insert(TypeId::of::<M>(), Cell::new(0.0));
    }

    pub fn module<M: EngineModule>(&self) -> OwnedRef<dyn EngineModule, M> {
        let module = self.modules.get(&TypeId::of::<M>()).unwrap().clone();

        OwnedRef::map(module.borrow_owned(), |v| v.as_any().downcast_ref::<M>().unwrap())
    }

    pub fn module_mut<M: EngineModule>(&self) -> OwnedRefMut<dyn EngineModule, M> {
        let module = self.modules.get(&TypeId::of::<M>()).unwrap().clone();

        OwnedRefMut::map(module.borrow_mut_owned(), |v| {
            v.as_mut_any().downcast_mut::<M>().unwrap()
        })
    }

    pub fn module_last_update_time<M: EngineModule>(&self) -> f64 {
        self.last_module_update_times[&TypeId::of::<M>()].get()
    }

    pub(crate) fn modules(&self) -> impl Iterator<Item = (&TypeId, &Lrc<dyn EngineModule>)> {
        self.modules.iter().rev()
    }

    #[inline]
    pub(crate) fn on_start(&self, ctx: &EngineContext) {
        for (_, module) in self.modules() {
            module.borrow_mut().on_start(ctx);
        }
    }

    #[inline]
    pub(crate) fn update_module(&self, module_id: &TypeId, dt: f64, ctx: &EngineContext) {
        let module = &self.modules[module_id];

        let t0 = Instant::now();
        module.borrow_mut().on_update(dt, ctx);
        let t1 = Instant::now();

        let time = (t1 - t0).as_secs_f64();
        self.last_module_update_times[module_id].set(time);
    }

    #[inline]
    pub(crate) fn on_wsi_event(&self, window: &Window, event: &WSIEvent, ctx: &EngineContext) {
        for (_, module) in self.modules() {
            module.borrow_mut().on_wsi_event(window, event, ctx);
        }
    }
}
