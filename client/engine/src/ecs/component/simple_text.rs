use crate::ecs::component::render_config::RenderLayer;
use crate::module::main_renderer::MaterialPipelineId;
use crate::module::ui::color::Color;

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

#[derive(Copy, Clone, PartialEq)]
pub struct TextStyle {
    /// Font registered in Renderer
    font_id: u16,
    /// Size in world units
    font_size: f32,
    font_style: FontStyle,
    color: Color,
}

impl TextStyle {
    pub fn new() -> Self {
        Self::default()
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

    pub fn with_color(mut self, color: Color) -> Self {
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

    pub fn color(&self) -> &Color {
        &self.color
    }

    pub fn color_mut(&mut self) -> &mut Color {
        &mut self.color
    }
}

impl Default for TextStyle {
    fn default() -> Self {
        Self {
            font_id: 0,
            font_size: 1.0,
            font_style: FontStyle::Normal,
            color: Color::grayscale(1.0),
        }
    }
}

#[derive(Clone, Default, PartialEq)]
pub struct StyledString {
    data: String,
    text_style: TextStyle,
}

impl StyledString {
    pub fn new(data: impl Into<String>, text_style: TextStyle) -> Self {
        Self {
            data: data.into(),
            text_style,
        }
    }

    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.text_style = style;
        self
    }

    pub fn data(&self) -> &str {
        &self.data
    }

    pub fn style(&self) -> &TextStyle {
        &self.text_style
    }

    pub fn style_mut(&mut self) -> &mut TextStyle {
        &mut self.text_style
    }
}

#[repr(u8)]
pub enum TextOverflow {
    Wrap,
}

#[derive(Copy, Clone, PartialEq, Eq)]
#[repr(u8)]
pub enum TextHAlign {
    Left,
    Center,
    Right,
}

impl Default for TextHAlign {
    fn default() -> Self {
        Self::Left
    }
}

#[derive(Clone)]
pub struct SimpleTextC {
    pub text: StyledString,
    pub h_align: TextHAlign,
    pub long_word_breaking: bool,
    pub max_width: f32,
    pub max_height: f32,
    pub render_type: RenderLayer,
    pub mat_pipeline: MaterialPipelineId,
}

impl SimpleTextC {
    pub fn new(mat_pipeline_id: MaterialPipelineId) -> Self {
        Self {
            text: Default::default(),
            h_align: Default::default(),
            long_word_breaking: false,
            max_width: f32::INFINITY,
            max_height: f32::INFINITY,
            render_type: RenderLayer::Main,
            mat_pipeline: mat_pipeline_id,
        }
    }

    pub fn with_text(mut self, text: StyledString) -> Self {
        self.text = text;
        self
    }

    pub fn with_render_type(mut self, stage: RenderLayer) -> Self {
        self.render_type = stage;
        self
    }

    pub fn with_h_align(mut self, align: TextHAlign) -> Self {
        self.h_align = align;
        self
    }

    pub fn with_long_word_breaking(mut self, do_break: bool) -> Self {
        self.long_word_breaking = do_break;
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
}
