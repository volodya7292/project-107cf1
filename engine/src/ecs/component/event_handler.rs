use crate::renderer::Renderer;
use entity_data::EntityId;

pub type OnAdded = fn(entity: &EntityId, renderer: &mut Renderer);

#[derive(Default)]
pub struct EventHandlerC {
    pub on_added: Option<OnAdded>,
}

pub trait EventHandlerI {
    fn on_added(_entity: &EntityId, _renderer: &mut Renderer) {}
}

impl EventHandlerC {
    pub fn new<I: EventHandlerI>() -> Self {
        Self {
            on_added: Some(I::on_added),
        }
    }
}
