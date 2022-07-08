use crate::ecs::scene::Scene;
use parking_lot::Mutex;
use std::any::Any;
use std::sync::Arc;
use vk_wrapper::CmdList;

pub mod text_renderer;

pub trait RendererModule {
    /// The returned CmdList will be queued after this method.
    fn on_update(&mut self, _scene: &mut Scene) -> Option<Arc<Mutex<CmdList>>> {
        None
    }
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}
