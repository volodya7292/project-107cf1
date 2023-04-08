use crate::module::scene::Scene;
use crate::EngineContext;
use common::glm::Vec2;
use common::types::IndexSet;
use entity_data::EntityId;
use std::hash::{Hash, Hasher};

pub type Factor = f32;

#[derive(Copy, Clone, PartialEq)]
pub enum Position {
    Auto,
    Relative(Vec2),
}

impl Default for Position {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Sizing {
    /// Uses size as close size as possible to the specified one.
    /// Elements of size [Self::Preferred] are expanded before [Self::Grow].
    Preferred(f32),
    /// The size is proportional to all siblings. Expanded after [Self::Preferred].
    Grow(Factor),
    /// Minimum possible size.
    FitContent,
}

impl Default for Sizing {
    fn default() -> Self {
        Self::FitContent
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Constraint {
    pub min: f32,
    pub max: f32,
}

impl Constraint {
    pub fn new(min: f32, max: f32) -> Self {
        Self { min, max }
    }

    pub fn exact(exact: f32) -> Self {
        Self::new(exact, exact)
    }

    pub fn clamp(&self, v: f32) -> f32 {
        v.clamp(self.min, self.max)
    }
}

impl Default for Constraint {
    fn default() -> Self {
        Self {
            min: 0.0,
            max: f32::INFINITY,
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct Padding {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

impl Padding {
    pub(crate) fn size(&self) -> Vec2 {
        Vec2::new(self.left + self.right, self.top + self.bottom)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum ContentFlow {
    Horizontal,
    Vertical,
}

impl ContentFlow {
    pub const fn axis(self) -> usize {
        match self {
            ContentFlow::Horizontal => 0,
            ContentFlow::Vertical => 1,
        }
    }

    pub const fn cross_axis(self) -> usize {
        (self.axis() + 1) & 1
    }
}

impl Default for ContentFlow {
    fn default() -> Self {
        Self::Vertical
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum FlowAlign {
    Start,
    Center,
    End,
}

impl Default for FlowAlign {
    fn default() -> Self {
        Self::Start
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum CrossAlign {
    Start,
    Center,
    End,
}

impl Default for CrossAlign {
    fn default() -> Self {
        Self::Start
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Overflow {
    Visible,
    Hidden,
}

impl Default for Overflow {
    fn default() -> Self {
        Self::Visible
    }
}

#[derive(Default)]
pub struct UILayoutC {
    pub position: Position,
    pub sizing: [Sizing; 2],
    pub constraints: [Constraint; 2],
    pub overflow: [Overflow; 2],
    pub align: CrossAlign,
    pub padding: Padding,
    pub content_flow: ContentFlow,
    pub flow_align: FlowAlign,
    pub shader_inverted_y: bool,
}

impl UILayoutC {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn row() -> Self {
        Self {
            content_flow: ContentFlow::Horizontal,
            ..Default::default()
        }
    }

    pub fn column() -> Self {
        Self {
            content_flow: ContentFlow::Vertical,
            ..Default::default()
        }
    }

    pub fn with_align(mut self, align: CrossAlign) -> Self {
        self.align = align;
        self
    }

    pub fn with_width(mut self, width: Sizing) -> Self {
        self.sizing[0] = width;
        self
    }

    pub fn with_height(mut self, height: Sizing) -> Self {
        self.sizing[1] = height;
        self
    }

    pub fn with_min_width(mut self, min_width: f32) -> Self {
        self.constraints[0].min = min_width;
        self
    }

    pub fn with_min_height(mut self, min_height: f32) -> Self {
        self.constraints[1].min = min_height;
        self
    }

    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.constraints[0].max = max_width;
        self
    }

    pub fn with_max_height(mut self, max_height: f32) -> Self {
        self.constraints[1].max = max_height;
        self
    }

    pub fn with_shader_inverted_y(mut self, enabled: bool) -> Self {
        self.shader_inverted_y = enabled;
        self
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct Rect {
    pub min: Vec2,
    pub max: Vec2,
}

impl Rect {
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: self.min.sup(&other.min),
            max: self.max.inf(&other.max),
        }
    }

    pub fn size_by_axis(&self, axis: usize) -> f32 {
        self.max[axis] - self.min[axis]
    }

    pub fn set_axis_size(&mut self, axis: usize, size: f32) {
        self.max[axis] = self.min[axis] + size;
    }

    pub fn contains_point(&self, point: &Vec2) -> bool {
        point >= &self.min && point < &self.max
    }
}

#[derive(Debug, Default, Copy, Clone)]
#[repr(C)]
pub struct RectUniformData {
    pub min: Vec2,
    pub max: Vec2,
}

#[derive(Default, Copy, Clone)]
pub struct UILayoutCacheC {
    pub(crate) final_min_size: Vec2,
    pub(crate) final_size: Vec2,
    pub(crate) relative_position: Vec2,
    pub(crate) global_position: Vec2,
    pub(crate) clip_rect: Rect,
    pub(crate) calculated_clip_rect: RectUniformData,
}

impl UILayoutCacheC {
    /// Returns final clipping rectangle in normalized coordinates.
    pub fn calculated_clip_rect(&self) -> &RectUniformData {
        &self.calculated_clip_rect
    }
}

type BasicEventCallback = fn(&EntityId, &mut Scene, &EngineContext);

#[derive(Copy, Clone)]
pub struct BasicEventCallbackWrapper(pub fn(&EntityId, &mut Scene, &EngineContext));

impl PartialEq for BasicEventCallbackWrapper {
    fn eq(&self, other: &Self) -> bool {
        let p1 = self.0 as *const u8;
        let p2 = other.0 as *const u8;
        p1 == p2
    }
}

impl Eq for BasicEventCallbackWrapper {}

impl Hash for BasicEventCallbackWrapper {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let p = self.0 as *const u8;
        p.hash(state);
    }
}

pub struct UIEventHandlerC {
    pub on_cursor_enter: IndexSet<BasicEventCallbackWrapper>,
    pub on_cursor_leave: IndexSet<BasicEventCallbackWrapper>,
    pub on_mouse_press: IndexSet<BasicEventCallbackWrapper>,
    pub on_mouse_release: IndexSet<BasicEventCallbackWrapper>,
    pub on_click: IndexSet<BasicEventCallbackWrapper>,
    pub enabled: bool,
}

impl Default for UIEventHandlerC {
    fn default() -> Self {
        Self {
            on_cursor_enter: Default::default(),
            on_cursor_leave: Default::default(),
            on_mouse_press: Default::default(),
            on_mouse_release: Default::default(),
            on_click: Default::default(),
            enabled: true,
        }
    }
}

pub trait UIEventHandlerI {
    fn on_cursor_enter(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_cursor_leave(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_mouse_press(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_mouse_release(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_click(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
}

impl UIEventHandlerI for () {}

impl UIEventHandlerC {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn passthrough() -> Self {
        Self {
            on_cursor_enter: Default::default(),
            on_cursor_leave: Default::default(),
            on_mouse_press: Default::default(),
            on_mouse_release: Default::default(),
            on_click: Default::default(),
            enabled: false,
        }
    }

    pub fn from_impl<I: UIEventHandlerI + 'static>() -> Self {
        Self {
            on_cursor_enter: IndexSet::from_iter([BasicEventCallbackWrapper(I::on_cursor_enter)]),
            on_cursor_leave: IndexSet::from_iter([BasicEventCallbackWrapper(I::on_cursor_enter)]),
            on_mouse_press: IndexSet::from_iter([BasicEventCallbackWrapper(I::on_cursor_enter)]),
            on_mouse_release: IndexSet::from_iter([BasicEventCallbackWrapper(I::on_cursor_enter)]),
            on_click: IndexSet::from_iter([BasicEventCallbackWrapper(I::on_cursor_enter)]),
            enabled: true,
        }
    }

    pub fn add_on_cursor_enter(mut self, handler: BasicEventCallback) -> Self {
        self.on_cursor_enter.insert(BasicEventCallbackWrapper(handler));
        self
    }

    pub fn add_on_cursor_leave(mut self, handler: BasicEventCallback) -> Self {
        self.on_cursor_leave.insert(BasicEventCallbackWrapper(handler));
        self
    }
}

macro_rules! ui_invoke_callback_set {
    ($callback_set: expr, $($param: expr),* $(,)?) => {
        for callback in &$callback_set {
            callback.0($($param,)*);
        }
    };
}
