pub mod main_renderer;
pub mod text_renderer;
pub mod ui_interaction_manager;
pub mod ui_renderer;

use crate::event::WSIEvent;
use crate::utils::wsi::WSISize;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRefMut};
use common::types::HashMap;
use entity_data::EntityId;
use std::any::{Any, TypeId};
use std::cell::{Ref, RefMut};

pub trait EngineModule: 'static {
    /// Called after an object has been added.
    fn on_object_added(&mut self, _: &EntityId) {}
    /// Called before an object is removed.
    fn on_object_remove(&mut self, _: &EntityId) {}
    /// Main loop
    fn on_update(&mut self) {}
    fn on_wsi_event(&mut self, _: &WSIEvent) {}

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

#[derive(Default)]
pub(crate) struct ModuleManager {
    modules: HashMap<TypeId, Lrc<dyn EngineModule>>,
}

impl ModuleManager {
    pub fn register_module<M: EngineModule>(&mut self, module: M) {
        self.modules.insert(TypeId::of::<M>(), Lrc::wrap(module));
    }

    pub fn module_mut<M: EngineModule>(&self) -> OwnedRefMut<dyn EngineModule, M> {
        let module = self.modules.get(&TypeId::of::<M>()).unwrap().clone();

        OwnedRefMut::map(module.borrow_mut_owned(), |v| {
            v.as_any_mut().downcast_mut::<M>().unwrap()
        })
    }

    #[inline]
    fn for_every<F: Fn(&mut dyn EngineModule)>(&self, f: F) {
        for module in self.modules.values() {
            let mut module = module.borrow_mut();
            f(&mut *module);
        }
    }

    #[inline]
    pub(crate) fn on_object_added(&self, id: &EntityId) {
        self.for_every(|module| {
            module.on_object_added(id);
        });
    }

    #[inline]
    pub(crate) fn on_object_remove(&self, id: &EntityId) {
        self.for_every(|module| {
            module.on_object_remove(id);
        });
    }

    #[inline]
    pub(crate) fn on_update(&self) {
        self.for_every(|module| {
            module.on_update();
        });
    }

    #[inline]
    pub(crate) fn on_wsi_event(&self, event: &WSIEvent) {
        self.for_every(|module| {
            module.on_wsi_event(event);
        });
    }
}
