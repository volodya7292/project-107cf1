pub mod container;
pub mod fancy_button;
pub mod image;
pub mod text;

use crate::game::MainApp;
use crate::rendering::ui::image::{ImageImpl, UIImage};
use crate::rendering::ui::text::{UIText, UITextImpl};
use ::image::ImageResult;
use common::lrc::OwnedRefMut;
use common::resource_file::{ResourceFile, ResourceRef};
use engine::ecs::component::ui::BasicEventCallback2;
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
    container::background::register_backgrounds(ctx);
    UIText::register(ctx);
    UIImage::register(ctx);
    fancy_button::register(ctx);
}

pub struct UICallbacks {
    pub interaction_enabled: bool,
    pub on_click: Option<Arc<dyn BasicEventCallback2<Output = ()>>>,
    pub on_cursor_enter: Option<Arc<dyn BasicEventCallback2<Output = ()>>>,
    pub on_cursor_leave: Option<Arc<dyn BasicEventCallback2<Output = ()>>>,
}

impl UICallbacks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_on_click(mut self, on_click: Arc<dyn BasicEventCallback2<Output = ()>>) -> Self {
        self.on_click = Some(on_click);
        self
    }

    pub fn with_on_cursor_enter(
        mut self,
        on_cursor_enter: Arc<dyn BasicEventCallback2<Output = ()>>,
    ) -> Self {
        self.on_cursor_enter = Some(on_cursor_enter);
        self
    }

    pub fn with_on_cursor_leave(
        mut self,
        on_cursor_leave: Arc<dyn BasicEventCallback2<Output = ()>>,
    ) -> Self {
        self.on_cursor_leave = Some(on_cursor_leave);
        self
    }
}

impl Default for UICallbacks {
    fn default() -> Self {
        Self {
            interaction_enabled: true,
            on_click: None,
            on_cursor_enter: None,
            on_cursor_leave: None,
        }
    }
}
