use crate::module::scene::Scene;
use crate::EngineContext;
use common::glm::Vec2;
use entity_data::EntityId;

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
#[derive(Copy, Clone, Eq, PartialEq)]
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
    pub uniform_crop_rect_offset: u32,
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

    pub fn with_uniform_crop_rect_offset(mut self, uniform_crop_rect_offset: u32) -> Self {
        self.uniform_crop_rect_offset = uniform_crop_rect_offset;
        self
    }
}

#[derive(Default, Copy, Clone, PartialEq)]
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

#[derive(Default, Copy, Clone)]
pub struct UILayoutCacheC {
    pub(crate) final_min_size: Vec2,
    pub(crate) final_size: Vec2,
    pub(crate) relative_position: Vec2,
    pub(crate) global_position: Vec2,
    pub(crate) clip_rect: Rect,
}

type BasicEventCallback = fn(entity: &EntityId, scene: &mut Scene, scene: &EngineContext);

#[derive(Copy, Clone)]
pub struct UIEventHandlerC {
    pub on_cursor_enter: BasicEventCallback,
    pub on_cursor_leave: BasicEventCallback,
    pub on_mouse_press: BasicEventCallback,
    pub on_mouse_release: BasicEventCallback,
    pub on_click: BasicEventCallback,
}

pub trait UIEventHandlerI {
    fn on_hover_enter(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_hover_exit(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_mouse_press(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_mouse_release(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
    fn on_click(_: &EntityId, _: &mut Scene, _: &EngineContext) {}
}

impl UIEventHandlerI for () {}

impl UIEventHandlerC {
    pub fn new<I: UIEventHandlerI>() -> Self {
        Self {
            on_cursor_enter: I::on_hover_enter,
            on_cursor_leave: I::on_hover_exit,
            on_mouse_press: I::on_mouse_press,
            on_mouse_release: I::on_mouse_release,
            on_click: I::on_click,
        }
    }
}

impl Default for UIEventHandlerC {
    fn default() -> Self {
        Self::new::<()>()
    }
}
