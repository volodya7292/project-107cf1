pub mod container;
pub mod fancy_button;
pub mod image;
pub mod text;

use crate::rendering::ui::image::{ImageImpl, UIImage};
use crate::rendering::ui::text::{UIText, UITextImpl};
use common::lrc::OwnedRefMut;
use engine::module::scene::Scene;
use engine::module::EngineModule;
use engine::EngineContext;

pub struct UIContext<'a> {
    scene: OwnedRefMut<dyn EngineModule, Scene>,
    ctx: &'a EngineContext<'a>,
}

impl<'a> UIContext<'a> {
    pub fn new(ctx: &'a EngineContext) -> Self {
        Self {
            scene: ctx.module_mut::<Scene>(),
            ctx,
        }
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut *self.scene
    }
}

pub fn register_ui_elements(ctx: &EngineContext) {
    UIText::register(ctx);
    UIImage::register(ctx);
}
