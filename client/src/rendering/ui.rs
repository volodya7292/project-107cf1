pub mod container;
pub mod fancy_button;
pub mod image;
pub mod text;

use crate::rendering::ui::container::{Container, ContainerImpl};
use crate::rendering::ui::fancy_button::{FancyButton, FancyButtonImpl};
use crate::rendering::ui::image::{ImageImpl, UIImage};
use crate::rendering::ui::text::{UIText, UITextImpl};
use ::image::ImageResult;
use common::lrc::OwnedRefMut;
use common::resource_file::{ResourceFile, ResourceRef};
use engine::module::scene::Scene;
use engine::module::EngineModule;
use engine::EngineContext;
use entity_data::StaticArchetype;
use std::ops::Deref;
use std::sync::Arc;

pub struct UIContext<'a> {
    scene: OwnedRefMut<dyn EngineModule, Scene>,
    ctx: &'a EngineContext<'a>,
    resources: Arc<ResourceFile>,
}

impl<'a> UIContext<'a> {
    pub fn new(ctx: &'a EngineContext, resources: &Arc<ResourceFile>) -> Self {
        Self {
            scene: ctx.module_mut::<Scene>(),
            ctx,
            resources: Arc::clone(resources),
        }
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut *self.scene
    }

    pub fn resource(&mut self, filename: &str) -> Result<ResourceRef, common::resource_file::Error> {
        self.resources.get(filename)
    }

    pub fn resource_image(
        &mut self,
        filename: &str,
    ) -> Result<ImageResult<::image::RgbaImage>, common::resource_file::Error> {
        let res = self.resources.get(filename)?;
        let dyn_img = ::image::load_from_memory(&res.read()?);
        Ok(dyn_img.map(|img| img.into_rgba8()))
    }
}

pub fn register_ui_elements(ctx: &EngineContext) {
    Container::register(ctx);
    UIText::register(ctx);
    UIImage::register(ctx);
    FancyButton::register(ctx);
}
