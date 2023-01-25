use nalgebra_glm::Vec2;
use std::sync::atomic::AtomicUsize;

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

#[derive(Copy, Clone)]
pub enum Sizing {
    Exact(f32),
    Preferred(f32),
    Grow(Factor),
    FitContent,
}

impl Default for Sizing {
    fn default() -> Self {
        Self::FitContent
    }
}

#[derive(Default, Copy, Clone)]
pub struct Padding {
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
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
    // SpaceBetween,
}

impl Default for FlowAlign {
    fn default() -> Self {
        Self::Start
    }
}

// #[repr(u8)]
// #[derive(Copy, Clone, Eq, PartialEq)]
// pub enum CrossFlowAlign {
//     Start,
//     Center,
//     End,
//     SpaceBetween,
// }

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
pub struct UILayout {
    position: Position,
    pub sizing: [Sizing; 2],
    self_cross_align: CrossAlign,
    padding: Padding,
    overflow: Overflow,
    content_flow: ContentFlow,
    flow_align: FlowAlign,
}

impl UILayout {
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

    pub fn with_width(mut self, width: Sizing) -> Self {
        self.sizing[0] = width;
        self
    }

    pub fn with_height(mut self, height: Sizing) -> Self {
        self.sizing[1] = height;
        self
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn padding(&self) -> &Padding {
        &self.padding
    }

    pub fn self_cross_align(&self) -> CrossAlign {
        self.self_cross_align
    }

    pub fn overflow(&self) -> Overflow {
        self.overflow
    }

    pub fn content_flow(&self) -> ContentFlow {
        self.content_flow
    }

    pub fn flow_align(&self) -> FlowAlign {
        self.flow_align
    }
}

#[derive(Default)]
pub struct UILayoutCache {
    pub(crate) intrinsic_min_size: Vec2,
    pub(crate) final_size: Vec2,
    pub(crate) relative_position: Vec2,
    pub(crate) global_position: Vec2,
}
