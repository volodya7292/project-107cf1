use nalgebra_glm::DVec2;

pub type Factor = f64;

#[derive(Copy, Clone, PartialEq)]
pub enum Position {
    Auto,
    Relative(DVec2),
}

impl Default for Position {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Copy, Clone)]
pub enum Sizing {
    Exact(f64),
    Preferred(f64),
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
    left: f64,
    right: f64,
    top: f64,
    bottom: f64,
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
    SpaceBetween,
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
pub enum Alignment {
    TopLeft,
    TopCenter,
    TopRight,
    CenterLeft,
    Center,
    CenterRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Default for Alignment {
    fn default() -> Self {
        Self::TopLeft
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
    size: [Sizing; 2],
    self_align: Alignment,
    padding: Padding,
    overflow: Overflow,
    content_flow: ContentFlow,
    flow_align: FlowAlign,
    cross_flow_align: FlowAlign,
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

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn sizing(&self) -> &[Sizing; 2] {
        &self.size
    }

    pub fn content_flow(&self) -> ContentFlow {
        self.content_flow
    }
}

#[derive(Default)]
pub struct UILayoutCache {
    pub(crate) intrinsic_min_size: DVec2,
    // pub(crate) max_size_allowed_by_parent: DVec2,
    pub(crate) final_size: DVec2,
}
