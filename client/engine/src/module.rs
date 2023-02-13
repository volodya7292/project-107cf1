pub mod input;
pub mod main_renderer;
pub mod text_renderer;
pub mod ui;
pub mod ui_interaction_manager;

use crate::event::WSIEvent;
use crate::EngineContext;
use common::any::AsAny;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRef, OwnedRefMut};
use common::types::IndexMap;
use entity_data::EntityId;
use std::any::TypeId;
use winit::window::Window;

pub trait EngineModule: AsAny {
    /// Called after an object has been added.
    fn on_object_added(&mut self, _: &EntityId, _: &EngineContext) {}
    /// Called before an object is removed.
    fn on_object_remove(&mut self, _: &EntityId, _: &EngineContext) {}
    /// Main loop
    fn on_update(&mut self, _: &EngineContext) {}
    fn on_wsi_event(&mut self, _: &Window, _: &WSIEvent, _: &EngineContext) {}
}

#[derive(Default)]
pub(crate) struct ModuleManager {
    modules: IndexMap<TypeId, Lrc<dyn EngineModule>>,
}

impl ModuleManager {
    /// Registers a new `EngineModule`.
    /// Callbacks to all modules will be called in reversed registration order of the modules.
    pub fn register_module<M: EngineModule>(&mut self, module: M) {
        self.modules.insert(TypeId::of::<M>(), Lrc::wrap(module));
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

    #[inline]
    fn for_every<F: Fn(&mut dyn EngineModule)>(&self, f: F) {
        for module in self.modules.values().rev() {
            let mut module = module.borrow_mut();
            f(&mut *module);
        }
    }

    #[inline]
    pub(crate) fn on_object_added(&self, id: &EntityId, ctx: &EngineContext) {
        self.for_every(|module| {
            module.on_object_added(id, ctx);
        });
    }

    #[inline]
    pub(crate) fn on_object_remove(&self, id: &EntityId, ctx: &EngineContext) {
        self.for_every(|module| {
            module.on_object_remove(id, ctx);
        });
    }

    #[inline]
    pub(crate) fn on_update(&self, ctx: &EngineContext) {
        self.for_every(|module| {
            module.on_update(ctx);
        });
    }

    #[inline]
    pub(crate) fn on_wsi_event(&self, window: &Window, event: &WSIEvent, ctx: &EngineContext) {
        self.for_every(|module| {
            module.on_wsi_event(window, event, ctx);
        });
    }
}
