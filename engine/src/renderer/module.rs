use crate::ecs::SceneAccess;
use entity_data::EntityId;
use parking_lot::Mutex;
use std::any::Any;
use std::sync::Arc;
use vk_wrapper::CmdList;

pub mod text_renderer;
pub mod ui_renderer;

pub trait RendererModule: Send + Sync {
    /// Note: entity rendering resources (e.g. vertex mesh) may be in use, destroy it only in `Self::on_update`.
    fn on_object_remove(&mut self, _id: &EntityId, _scene: SceneAccess<()>) {}
    /// The returned CmdList will be queued after this method.
    fn on_update(&mut self, _scene: SceneAccess<()>) -> Option<Arc<Mutex<CmdList>>> {
        None
    }
    fn on_resize(&mut self, _new_physical_size: (u32, u32), _scale_factor: f64) {}
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
