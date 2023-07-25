use common::glm::Vec4;

/// `r`,`g`,`b` are in range [0; inf]; `a` is in range [0; 1].
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color(Vec4);

impl Color {
    pub const TRANSPARENT: Color = Color::new(0.0, 0.0, 0.0, 0.0);
    pub const BLACK: Color = Color::grayscale(0.0);
    pub const WHITE: Color = Color::grayscale(1.0);

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(Vec4::new(r, g, b, a))
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    pub fn from_hex(hex: u32) -> Self {
        Self::new(
            (hex >> 24 & 0xff) as f32 / 255.0,
            (hex >> 16 & 0xff) as f32 / 255.0,
            (hex >> 8 & 0xff) as f32 / 255.0,
            (hex >> 0 & 0xff) as f32 / 255.0,
        )
    }

    pub const fn grayscale(v: f32) -> Self {
        Self::new(v, v, v, 1.0)
    }

    pub fn with_alpha(mut self, a: f32) -> Self {
        self.0.w = a;
        self
    }

    pub fn into_raw(self) -> Vec4 {
        self.0
    }
}

impl From<u32> for Color {
    fn from(value: u32) -> Self {
        Self::from_hex(value)
    }
}

impl From<Vec4> for Color {
    fn from(value: Vec4) -> Self {
        Self(value)
    }
}
