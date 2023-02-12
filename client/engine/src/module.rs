pub mod main_renderer;
pub mod text_renderer;
pub mod ui_interaction_manager;
pub mod ui_renderer;

use crate::event::Event;
use crate::utils::wsi::WSISize;
use entity_data::EntityId;
use std::any::Any;

pub trait EngineModule: 'static {
    /// Note: entity rendering resources (e.g. vertex mesh) may be in use, destroy it only in `Self::on_update`.
    fn on_object_added(&mut self, _id: &EntityId) {}
    /// Note: entity rendering resources (e.g. vertex mesh) may be in use, destroy it only in `Self::on_update`.
    fn on_object_remove(&mut self, _id: &EntityId) {}
    /// Main loop
    fn on_update(&mut self) {}
    fn on_resize(&mut self, _new_size: WSISize<u32>) {}
    fn on_event(&mut self, _: &Event) {}

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
