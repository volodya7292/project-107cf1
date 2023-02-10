pub mod text_renderer;
pub mod ui_interaction_manager;
pub mod ui_renderer;

use crate::ecs::SceneAccess;
use crate::event::Event;
use crate::renderer::RendererContext;
use crate::utils::wsi::WSISize;
use entity_data::EntityId;
use parking_lot::Mutex;
use std::any::Any;
use std::sync::Arc;
use vk_wrapper::CmdList;

pub trait RendererModule: Send + Sync {
    /// Note: entity rendering resources (e.g. vertex mesh) may be in use, destroy it only in `Self::on_update`.
    fn on_object_remove(&mut self, _id: &EntityId, _scene: SceneAccess<()>) {}
    /// The returned CmdList will be queued after this method.
    fn on_update(&mut self, _scene: SceneAccess<()>) -> Option<Arc<Mutex<CmdList>>> {
        None
    }
    fn on_resize(&mut self, _new_size: WSISize<u32>) {}
    fn on_event(&mut self, _scene: SceneAccess<RendererContext>, _: &Event) {}

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
