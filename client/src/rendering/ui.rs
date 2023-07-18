pub mod container;
pub mod fancy_button;
pub mod image;
pub mod text;

use crate::game::MainApp;
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
use std::sync::Arc;

pub const STATE_ENTITY_ID: &'static str = "__entity_id";

pub struct UIResources(pub Arc<ResourceFile>);

pub struct UIContext<'a> {
    scene: OwnedRefMut<dyn EngineModule, Scene>,
    ctx: EngineContext<'a>,
    resources: Arc<ResourceFile>,
}

impl<'a> UIContext<'a> {
    pub fn new(ctx: EngineContext<'a>) -> Self {
        let scene = ctx.module_mut::<Scene>();
        let resources = Arc::clone(&scene.resource::<UIResources>().0);
        Self {
            scene,
            ctx,
            resources,
        }
    }

    pub fn ctx(&self) -> &EngineContext {
        &self.ctx
    }

    pub fn app(&mut self) -> OwnedRefMut<dyn EngineModule, MainApp> {
        self.ctx.module_mut::<MainApp>()
    }

    pub fn scene(&mut self) -> &mut Scene {
        &mut *self.scene
    }

    pub fn resource(&mut self, filename: &str) -> Result<ResourceRef, common::resource_file::Error> {
        self.resources.get(filename)
    }

    pub fn resource_image(
        resources: &Arc<ResourceFile>,
        filename: &str,
    ) -> Result<ImageResult<::image::RgbaImage>, common::resource_file::Error> {
        let res = resources.get(filename)?;
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
