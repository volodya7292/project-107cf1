use crate::EngineContext;
use entity_data::EntityId;

pub trait UIState: Send + Sync + 'static {
    fn on_update(_entity: &EntityId, _ctx: &EngineContext, _dt: f64) {}
}

impl UIState for () {}
