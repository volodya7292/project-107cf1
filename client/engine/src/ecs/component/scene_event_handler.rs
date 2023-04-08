use crate::module::scene::Scene;
use crate::EngineContext;
use common::types::HashMap;
use entity_data::{Component, EntityId};
use std::any::TypeId;

type BasicEventCallback = fn(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext, dt: f64);

type ComponentChangedCallback = fn(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext);

#[derive(Default, Clone)]
pub struct SceneEventHandler {
    pub on_update: Option<BasicEventCallback>,
    pub on_component_update: HashMap<TypeId, ComponentChangedCallback>,
}

impl SceneEventHandler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_on_update(mut self, on_update: BasicEventCallback) -> Self {
        self.on_update = Some(on_update);
        self
    }

    pub fn with_on_component_update<C: Component>(mut self, on_update: ComponentChangedCallback) -> Self {
        self.on_component_update.insert(TypeId::of::<C>(), on_update);
        self
    }

    #[inline]
    pub fn on_update(&self) -> BasicEventCallback {
        self.on_update.unwrap_or(|_, _, _, _| {})
    }

    #[inline]
    pub fn on_component_update(&self, ty: &TypeId) -> Option<&ComponentChangedCallback> {
        self.on_component_update.get(ty)
    }
}
