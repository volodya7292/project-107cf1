use crate::EngineContext;
use common::types::HashMap;
use entity_data::{Component, EntityId};
use std::any::TypeId;

pub type OnUpdateCallback = fn(entity: &EntityId, ctx: &EngineContext, dt: f64);

type ComponentChangedCallback = fn(entity: &EntityId, ctx: &EngineContext);

#[derive(Default, Clone)]
pub struct SceneEventHandler {
    pub on_component_update: HashMap<TypeId, ComponentChangedCallback>,
}

impl SceneEventHandler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_on_component_update<C: Component>(mut self, on_update: ComponentChangedCallback) -> Self {
        self.on_component_update.insert(TypeId::of::<C>(), on_update);
        self
    }

    pub fn set_handler<C: Component>(&mut self, on_update: ComponentChangedCallback) {
        self.on_component_update.insert(TypeId::of::<C>(), on_update);
    }

    #[inline]
    pub fn on_component_update(&self, ty: &TypeId) -> Option<&ComponentChangedCallback> {
        self.on_component_update.get(ty)
    }
}
