use nalgebra_glm::Vec4;
use smallvec::SmallVec;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum FontStyle {
    Normal = 0,
    Italic = 1,
}

impl FontStyle {
    pub const fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Normal,
            1 => Self::Italic,
            _ => panic!("Incorrect FontStyle value"),
        }
    }
}

#[derive(Clone)]
pub struct FormattedString {
    data: String,
    /// Font registered in Renderer
    font_id: u32,
    /// Size in world units
    font_size: f32,
    font_style: FontStyle,
    color: Vec4,
}

impl FormattedString {
    pub fn new(data: String) -> Self {
        Self {
            data,
            font_id: u32::MAX,
            font_size: 1.0,
            font_style: FontStyle::Normal,
            color: Vec4::from_element(1.0),
        }
    }

    pub fn with_font(mut self, font_id: u32) -> Self {
        self.font_id = font_id;
        self
    }

    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    pub fn with_color(mut self, color: Vec4) -> Self {
        self.color = color;
        self
    }
}

#[derive(Clone)]
pub struct TextBlock {
    strings: SmallVec<[FormattedString; 2]>,
    max_width: f32,
    max_height: f32,
}

impl TextBlock {
    pub fn new(strings: &[FormattedString]) -> Self {
        Self {
            strings: strings.into(),
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
        }
    }

    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_width = max_width;
        self
    }

    pub fn with_max_height(mut self, max_height: f32) -> Self {
        self.max_height = max_height;
        self
    }
}

pub struct RawText {
    block: TextBlock,
}

impl RawText {
    pub fn new(block: TextBlock) -> Self {
        Self { block }
    }
}
