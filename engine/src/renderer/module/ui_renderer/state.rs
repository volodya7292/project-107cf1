use crate::renderer::Renderer;
use entity_data::EntityId;

pub trait UIState: Send + Sync + 'static {
    // TODO: reimplement this in a more usable way
    // fn on_update(_entity: &EntityId, _renderer: &mut Renderer) {}
}

impl UIState for () {}
