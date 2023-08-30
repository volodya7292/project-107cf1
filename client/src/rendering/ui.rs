pub mod backgrounds;
pub mod container;
pub mod fancy_button;
pub mod fancy_text_input;
pub mod image;
pub mod scrollable_container;
pub mod text;
pub mod text_input;

use crate::rendering::ui::image::{ImageImpl, UIImage};
use crate::rendering::ui::text::{UIText, UITextImpl};
use engine::ecs::component::ui::{
    BasicEventCallback, ClickedCallback, KeyboardCallback, ScrollCallback, SizeUpdateCallback,
    UIEventHandlerC,
};
use engine::EngineContext;

pub const STATE_ENTITY_ID: &str = "__entity_id";
pub const LOCAL_VAR_OPACITY: &str = "__opacity";

pub fn register_ui_elements(ctx: &EngineContext) {
    container::background::register_backgrounds(ctx);
    UIText::register(ctx);
    UIImage::register(ctx);
    backgrounds::register(ctx);
}

#[derive(Clone)]
pub struct UICallbacks {
    pub interaction_enabled: bool,
    pub focusable: bool,
    pub autofocus: bool,
    pub on_click: Option<ClickedCallback>,
    pub on_cursor_enter: Option<BasicEventCallback>,
    pub on_cursor_leave: Option<BasicEventCallback>,
    pub on_scroll: Option<ScrollCallback>,
    pub on_focus_in: Option<BasicEventCallback>,
    pub on_focus_out: Option<BasicEventCallback>,
    pub on_keyboard: Option<KeyboardCallback>,
    pub on_size_update: Option<SizeUpdateCallback>,
}

impl UICallbacks {
    pub fn interaction(mut self, enabled: bool) -> Self {
        self.interaction_enabled = enabled;
        self
    }

    pub fn focusable(mut self, focusable: bool) -> Self {
        self.focusable = focusable;
        self
    }

    pub fn autofocus(mut self, autofocus: bool) -> Self {
        self.autofocus = autofocus;
        self
    }

    pub fn on_click(mut self, on_click: ClickedCallback) -> Self {
        self.on_click = Some(on_click);
        self
    }

    pub fn on_cursor_enter(mut self, on_cursor_enter: BasicEventCallback) -> Self {
        self.on_cursor_enter = Some(on_cursor_enter);
        self
    }

    pub fn on_cursor_leave(mut self, on_cursor_leave: BasicEventCallback) -> Self {
        self.on_cursor_leave = Some(on_cursor_leave);
        self
    }

    pub fn on_scroll(mut self, on_scroll: ScrollCallback) -> Self {
        self.on_scroll = Some(on_scroll);
        self
    }

    pub fn on_focus_in(mut self, on_focus_in: BasicEventCallback) -> Self {
        self.on_focus_in = Some(on_focus_in);
        self
    }

    pub fn on_focus_out(mut self, on_focus_out: BasicEventCallback) -> Self {
        self.on_focus_out = Some(on_focus_out);
        self
    }

    pub fn on_keyboard(mut self, on_keyboard: KeyboardCallback) -> Self {
        self.on_keyboard = Some(on_keyboard);
        self
    }

    pub fn on_size_update(mut self, on_size_update: SizeUpdateCallback) -> Self {
        self.on_size_update = Some(on_size_update);
        self
    }
}

impl Default for UICallbacks {
    fn default() -> Self {
        Self {
            interaction_enabled: true,
            focusable: false,
            autofocus: false,
            on_click: None,
            on_cursor_enter: None,
            on_cursor_leave: None,
            on_scroll: None,
            on_focus_in: None,
            on_focus_out: None,
            on_keyboard: None,
            on_size_update: None,
        }
    }
}

impl PartialEq for UICallbacks {
    fn eq(&self, other: &Self) -> bool {
        self.interaction_enabled == other.interaction_enabled
            && self.focusable == other.focusable
            && self.autofocus == other.autofocus
    }
}

impl From<UICallbacks> for UIEventHandlerC {
    fn from(value: UICallbacks) -> Self {
        Self {
            enabled: value.interaction_enabled,
            focusable: value.focusable,
            autofocus: value.autofocus,
            on_click: value.on_click,
            on_cursor_enter: value.on_cursor_enter,
            on_cursor_leave: value.on_cursor_leave,
            on_mouse_press: None,
            on_mouse_release: None,
            on_scroll: value.on_scroll,
            on_keyboard: value.on_keyboard,
            on_focus_in: value.on_focus_in,
            on_focus_out: value.on_focus_out,
            on_size_update: value.on_size_update,
        }
    }
}

pub fn ui_callbacks() -> UICallbacks {
    UICallbacks::default()
}
