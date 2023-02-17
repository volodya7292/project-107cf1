use crate::module::scene::Scene;
use crate::EngineContext;
use entity_data::EntityId;

type BasicEventCallback = fn(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext);

#[derive(Default, Copy, Clone)]
pub struct SceneEventHandler {
    pub on_update: Option<BasicEventCallback>,
    pub on_update_active: bool,
}

impl SceneEventHandler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_on_update(mut self, on_update: BasicEventCallback) -> Self {
        self.on_update = Some(on_update);
        self
    }

    pub fn with_on_update_active(mut self, active: bool) -> Self {
        self.on_update_active = active;
        self
    }

    #[inline]
    pub fn on_update(&self) -> BasicEventCallback {
        self.on_update.unwrap_or(|_, _, _| {})
    }

    pub fn set_on_update_active(&mut self, active: bool) {
        self.on_update_active = active;
    }
}
