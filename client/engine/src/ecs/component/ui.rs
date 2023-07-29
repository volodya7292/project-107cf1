use crate::event::WSIKeyboardInput;
use crate::module::scene::EntityAccess;
use crate::utils::transition::AnimatedValue;
use crate::EngineContext;
use common::glm::Vec2;
use entity_data::EntityId;
use std::sync::Arc;

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
    /// Minimum size that fits tightly interior contents.
    FitContent,
    /// Uses a size as close as possible to the specified one.
    /// Elements of size [Self::Preferred] are expanded before [Self::Grow].
    Preferred(f32),
    /// The size is proportional to all siblings.
    Grow(Factor),
    ParentBased(fn(&EntityAccess<()>, &EngineContext, parent_size: &Vec2) -> f32),
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
    pub fn equal(value: f32) -> Self {
        Self {
            left: value,
            right: value,
            top: value,
            bottom: value,
        }
    }

    pub fn hv(horizontal: f32, vertical: f32) -> Self {
        Self {
            left: horizontal,
            right: horizontal,
            top: vertical,
            bottom: vertical,
        }
    }

    pub fn size(&self) -> Vec2 {
        Vec2::new(self.left + self.right, self.top + self.bottom)
    }
}

#[repr(u8)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
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

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Visibility {
    Opacity(AnimatedValue<f32>),
    Collapsed,
}

impl Visibility {
    pub fn visible() -> Self {
        Self::Opacity(1.0.into())
    }

    pub fn hidden() -> Self {
        Self::Opacity(0.0.into())
    }

    pub fn opacity(&self) -> f32 {
        match self {
            Visibility::Opacity(opacity) => *opacity.current(),
            Visibility::Collapsed => 0.0,
        }
    }

    pub fn is_visible(&self) -> bool {
        match self {
            Visibility::Opacity(opacity) if *opacity.current() < 0.001 => false,
            Visibility::Opacity(_) => true,
            Visibility::Collapsed => false,
        }
    }

    pub fn as_opacity_mut(&mut self) -> &mut AnimatedValue<f32> {
        match self {
            Visibility::Opacity(opacity) => opacity,
            Visibility::Collapsed => {
                panic!("Invalid value");
            }
        }
    }
}

impl Default for Visibility {
    fn default() -> Self {
        Self::visible()
    }
}

#[derive(Copy, Clone)]
pub struct UITransform {
    pub offset: Vec2,
}

impl UITransform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_offset(mut self, offset: Vec2) -> Self {
        self.offset = offset;
        self
    }
}

impl Default for UITransform {
    fn default() -> Self {
        Self {
            offset: Vec2::from_element(0.0),
        }
    }
}

#[derive(Default, Copy, Clone)]
pub struct UILayoutC {
    pub position: Position,
    pub sizing: [Sizing; 2],
    pub constraints: [Constraint; 2],
    pub overflow: [Overflow; 2],
    pub content_transform: UITransform,
    pub align: CrossAlign,
    pub padding: Padding,
    pub content_flow: ContentFlow,
    pub flow_align: FlowAlign,
    pub shader_inverted_y: bool,
    pub visibility: Visibility,
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

    pub fn with_position(mut self, position: Position) -> Self {
        self.position = position;
        self
    }

    pub fn with_content_transform(mut self, transform: UITransform) -> Self {
        self.content_transform = transform;
        self
    }

    pub fn with_align(mut self, align: CrossAlign) -> Self {
        self.align = align;
        self
    }

    pub fn with_padding(mut self, padding: Padding) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_width(mut self, width: Sizing) -> Self {
        self.sizing[0] = width;
        self
    }

    pub fn with_fixed_width(mut self, width: f32) -> Self {
        self.constraints[0] = Constraint::exact(width);
        self
    }

    pub fn with_fixed_height(mut self, height: f32) -> Self {
        self.constraints[1] = Constraint::exact(height);
        self
    }

    pub fn with_height(mut self, height: Sizing) -> Self {
        self.sizing[1] = height;
        self
    }

    pub fn with_width_grow(self) -> Self {
        self.with_width(Sizing::Grow(1.0))
    }

    pub fn with_height_grow(self) -> Self {
        self.with_height(Sizing::Grow(1.0))
    }

    pub fn with_grow(self) -> Self {
        self.with_width_grow().with_height_grow()
    }

    pub fn with_width_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints[0] = constraint;
        self
    }

    pub fn with_height_constraint(mut self, constraint: Constraint) -> Self {
        self.constraints[1] = constraint;
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

    pub fn with_visibility(mut self, visibility: Visibility) -> Self {
        self.visibility = visibility;
        self
    }

    pub fn with_shader_inverted_y(mut self, enabled: bool) -> Self {
        self.shader_inverted_y = enabled;
        self
    }
}

#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub struct ClipRect {
    pub min: Vec2,
    pub max: Vec2,
}

impl ClipRect {
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: self.min.sup(&other.min),
            max: self.max.inf(&other.max),
        }
    }

    pub fn size(&self) -> Vec2 {
        self.max - self.min
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
    pub(crate) clip_rect: ClipRect,
    pub(crate) calculated_clip_rect: RectUniformData,
}

impl UILayoutCacheC {
    /// Returns final clipping rectangle.
    pub fn clip_rect(&self) -> &ClipRect {
        &self.clip_rect
    }

    /// Returns final clipping rectangle in normalized coordinates.
    pub fn normalized_clip_rect(&self) -> &RectUniformData {
        &self.calculated_clip_rect
    }

    pub fn global_position(&self) -> &Vec2 {
        &self.global_position
    }

    pub fn final_size(&self) -> &Vec2 {
        &self.final_size
    }
}

#[macro_export]
macro_rules! define_callback {
    ($name: ident ($($params: ty $(,)?)*)) => {
        pub trait $name: Fn($($params,)*) + Send + Sync + 'static {}
        impl<F: Fn($($params,)*) + Send + Sync + 'static> $name for F {}
    };
}

define_callback!(BasicEventCallback(&EntityId, &EngineContext));
define_callback!(ClickedCallback(&EntityId, &EngineContext, Vec2));
define_callback!(KeyPressedCallback(&EntityId, &EngineContext, WSIKeyboardInput));

pub struct UIEventHandlerC {
    pub on_cursor_enter: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_cursor_leave: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_mouse_press: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_mouse_release: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_key_press: Option<Arc<dyn KeyPressedCallback<Output = ()>>>,
    pub on_focus_in: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_focus_out: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub on_click: Option<Arc<dyn ClickedCallback<Output = ()>>>,
    pub on_size_update: Option<Arc<dyn BasicEventCallback<Output = ()>>>,
    pub focusable: bool,
    pub enabled: bool,
}

impl Default for UIEventHandlerC {
    fn default() -> Self {
        Self {
            on_cursor_enter: None,
            on_cursor_leave: None,
            on_mouse_press: None,
            on_mouse_release: None,
            on_key_press: None,
            on_focus_in: None,
            on_focus_out: None,
            on_click: None,
            on_size_update: None,
            focusable: false,
            enabled: true,
        }
    }
}

impl UIEventHandlerC {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn passthrough() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    pub fn add_on_size_update(&mut self, handler: Arc<dyn BasicEventCallback<Output = ()>>) {
        self.on_size_update = Some(self.on_size_update.take().map_or_else(|| handler, |v| v));
    }
}
