use common::glm::{Vec3, Vec4};

/// An sRGB-color.
/// `r`,`g`,`b` are in range [0; inf]; `a` is in range [0; 1].
#[derive(Debug, Copy, Clone, Default, PartialEq)]
pub struct Color(Vec4);

impl Color {
    pub const TRANSPARENT: Color = Color::new(0.0, 0.0, 0.0, 0.0);
    pub const BLACK: Color = Color::grayscale(0.0);
    pub const WHITE: Color = Color::grayscale(1.0);
    pub const DARK_RED: Color = Color::rgb(0.7, 0.1, 0.1);

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(Vec4::new(r, g, b, a))
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    pub fn from_srgb(r: f32, g: f32, b: f32) -> Self {
        let rgb = Vec3::new(r, g, b).map(|v| if v < 1.0 { v.powf(2.2) } else { v });
        let rgba = rgb.push(1.0);
        Self(rgba)
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

    pub fn with_brightness(mut self, brightness: f32) -> Self {
        self.0.x *= brightness;
        self.0.y *= brightness;
        self.0.z *= brightness;
        self
    }

    pub fn alpha(&self) -> f32 {
        self.0.w
    }

    pub fn into_raw_linear(self) -> Vec4 {
        let rgb = self.0.xyz().map(|v| if v < 1.0 { v.powf(2.2) } else { v });
        rgb.push(self.alpha())
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
