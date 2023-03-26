use crate::module::scene::Scene;
use crate::EngineContext;
use entity_data::EntityId;

type BasicEventCallback = fn(entity: &EntityId, scene: &mut Scene, ctx: &EngineContext);

#[derive(Default, Copy, Clone)]
pub struct SceneEventHandler {
    pub on_update: Option<BasicEventCallback>,
}

impl SceneEventHandler {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_on_update(mut self, on_update: BasicEventCallback) -> Self {
        self.on_update = Some(on_update);
        self
    }

    #[inline]
    pub fn on_update(&self) -> BasicEventCallback {
        self.on_update.unwrap_or(|_, _, _| {})
    }
}
