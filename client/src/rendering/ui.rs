pub mod backgrounds;
pub mod container;
pub mod fancy_button;
pub mod fancy_text_input;
pub mod image;
pub mod text;
pub mod text_input;

use crate::game::{EngineCtxGameExt, MainApp};
use crate::rendering::ui::image::{ImageImpl, UIImage};
use crate::rendering::ui::text::{UIText, UITextImpl};
use ::image::ImageResult;
use common::lrc::OwnedRefMut;
use common::resource_file::BufferedResourceReader;
use engine::ecs::component::ui::{BasicEventCallback, ClickedCallback, KeyPressedCallback, UIEventHandlerC};
use engine::module::scene::Scene;
use engine::module::EngineModule;
use engine::EngineContext;
use std::sync::Arc;

pub const STATE_ENTITY_ID: &'static str = "__entity_id";

pub struct UIContext<'a> {
    scene: OwnedRefMut<dyn EngineModule, Scene>,
    ctx: EngineContext<'a>,
    resources: Arc<BufferedResourceReader>,
}

impl<'a> UIContext<'a> {
    pub fn new(ctx: EngineContext<'a>) -> Self {
        let scene = ctx.module_mut::<Scene>();
        let resources = Arc::clone(&scene.resource::<Arc<BufferedResourceReader>>());
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

    pub fn resource(&mut self, filename: &str) -> Result<Arc<[u8]>, common::resource_file::Error> {
        self.resources.get(filename)
    }

    pub fn resource_image(
        resources: &Arc<BufferedResourceReader>,
        filename: &str,
    ) -> Result<ImageResult<::image::RgbaImage>, common::resource_file::Error> {
        let res = resources.get(filename)?;
        let dyn_img = ::image::load_from_memory(&res);
        Ok(dyn_img.map(|img| img.into_rgba8()))
    }
}

pub fn register_ui_elements(ctx: &EngineContext) {
    container::background::register_backgrounds(ctx);
    UIText::register(ctx);
    UIImage::register(ctx);
    backgrounds::register(ctx);
}

pub struct UICallbacks {
    pub interaction_enabled: bool,
    pub focusable: bool,
    pub on_click: Option<Arc<dyn ClickedCallback<Output = ()>>>,
    pub on_cursor_enter: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_cursor_leave: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_focus_in: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_focus_out: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_key_press: Option<Arc<dyn KeyPressedCallback<Output = ()>>>,
    pub on_size_update: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
}

impl UICallbacks {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply_to_event_handler(&self, event_handler: &mut UIEventHandlerC) {
        event_handler.enabled = self.interaction_enabled;
        event_handler.focusable = self.focusable;
        event_handler.on_click = self.on_click.clone();
        event_handler.on_cursor_enter = self.on_cursor_enter.clone();
        event_handler.on_cursor_leave = self.on_cursor_leave.clone();
        event_handler.on_key_press = self.on_key_press.clone();
        event_handler.on_focus_in = self.on_focus_in.clone();
        event_handler.on_focus_out = self.on_focus_out.clone();
        event_handler.on_size_update = self.on_size_update.clone();
    }

    pub fn with_focusable(mut self, focusable: bool) -> Self {
        self.focusable = focusable;
        self
    }

    pub fn with_on_click(mut self, on_click: Arc<dyn ClickedCallback<Output = ()>>) -> Self {
        self.on_click = Some(on_click);
        self
    }

    pub fn with_on_cursor_enter(mut self, on_cursor_enter: Arc<dyn BasicEventCallback<Output = ()>>) -> Self {
        self.on_cursor_enter = Some(on_cursor_enter);
        self
    }

    pub fn with_on_cursor_leave(mut self, on_cursor_leave: Arc<dyn BasicEventCallback<Output = ()>>) -> Self {
        self.on_cursor_leave = Some(on_cursor_leave);
        self
    }

    pub fn with_on_focus_in(mut self, on_focus_in: Arc<dyn BasicEventCallback<Output = ()>>) -> Self {
        self.on_focus_in = Some(on_focus_in);
        self
    }

    pub fn with_on_focus_out(mut self, on_focus_out: Arc<dyn BasicEventCallback<Output = ()>>) -> Self {
        self.on_focus_out = Some(on_focus_out);
        self
    }

    pub fn with_on_key_press(mut self, on_key_press: Arc<dyn KeyPressedCallback<Output = ()>>) -> Self {
        self.on_key_press = Some(on_key_press);
        self
    }

    pub fn with_on_size_update(mut self, on_size_update: Arc<dyn BasicEventCallback<Output = ()>>) -> Self {
        self.on_size_update = Some(on_size_update);
        self
    }
}

impl Default for UICallbacks {
    fn default() -> Self {
        Self {
            interaction_enabled: true,
            focusable: false,
            on_click: None,
            on_cursor_enter: None,
            on_cursor_leave: None,
            on_focus_in: None,
            on_focus_out: None,
            on_key_press: None,
            on_size_update: None,
        }
    }
}
