use crate::renderer;
use entity_data::EntityId;
use parking_lot::Mutex;
use std::any::Any;
use std::sync::Arc;
use vk_wrapper::CmdList;

pub mod text_renderer;
pub mod ui_renderer;

pub trait RendererModule {
    /// Note: entity rendering resources (e.g. vertex mesh) may be in use, destroy it only in `Self::on_update`.
    fn on_object_remove(&mut self, _id: &EntityId, _scene: renderer::Internals) {}
    /// The returned CmdList will be queued after this method.
    fn on_update(&mut self, _internals: renderer::Internals) -> Option<Arc<Mutex<CmdList>>> {
        None
    }
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
