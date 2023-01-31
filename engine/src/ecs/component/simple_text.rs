use nalgebra_glm::U8Vec4;

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

#[derive(Copy, Clone)]
pub struct TextStyle {
    /// Font registered in Renderer
    font_id: u16,
    /// Size in world units
    font_size: f32,
    font_style: FontStyle,
    color: U8Vec4,
}

impl TextStyle {
    pub fn new() -> Self {
        Self {
            font_id: 0,
            font_size: 1.0,
            font_style: FontStyle::Normal,
            color: U8Vec4::from_element(255),
        }
    }

    pub fn with_font(mut self, font_id: u16) -> Self {
        self.font_id = font_id;
        self
    }

    pub fn with_font_size(mut self, font_size: f32) -> Self {
        self.font_size = font_size;
        self
    }

    pub fn with_font_style(mut self, style: FontStyle) -> Self {
        self.font_style = style;
        self
    }

    pub fn with_color(mut self, color: U8Vec4) -> Self {
        self.color = color;
        self
    }

    pub fn font_id(&self) -> u16 {
        self.font_id
    }

    pub fn font_size(&self) -> f32 {
        self.font_size
    }

    pub fn font_style(&self) -> FontStyle {
        self.font_style
    }

    pub fn color(&self) -> U8Vec4 {
        self.color
    }
}

#[derive(Clone)]
pub struct StyledString {
    data: String,
    text_style: TextStyle,
}

impl StyledString {
    pub fn new(data: String, text_style: TextStyle) -> Self {
        Self { data, text_style }
    }

    pub fn data(&self) -> &str {
        &self.data
    }

    pub fn style(&self) -> &TextStyle {
        &self.text_style
    }
}

#[repr(u8)]
pub enum TextOverflow {
    WRAP,
}

#[derive(Copy, Clone)]
#[repr(u8)]
pub enum TextHAlign {
    LEFT,
    CENTER,
    RIGHT,
}

#[derive(Clone)]
pub struct SimpleTextC {
    text: StyledString,
    h_align: TextHAlign,
    max_width: f32,
    max_height: f32,
}

impl SimpleTextC {
    pub fn new(text: StyledString) -> Self {
        Self {
            text,
            h_align: TextHAlign::LEFT,
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
        }
    }

    pub fn with_h_align(mut self, align: TextHAlign) -> Self {
        self.h_align = align;
        self
    }

    pub fn with_max_width(mut self, max_width: f32) -> Self {
        self.max_width = max_width;
        self
    }

    pub fn with_max_height(mut self, max_height: f32) -> Self {
        self.max_height = max_height;
        self
    }

    pub fn string(&self) -> &StyledString {
        &self.text
    }

    pub fn h_align(&self) -> TextHAlign {
        self.h_align
    }

    pub fn max_width(&self) -> f32 {
        self.max_width
    }

    pub fn max_height(&self) -> f32 {
        self.max_height
    }
}
