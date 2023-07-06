use common::glm::Vec4;

/// `r`,`g`,`b` are in range [0; inf]; `a` is in range [0; 1].
#[derive(Copy, Clone, Default)]
pub struct Color(Vec4);

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self(Vec4::new(r, g, b, a))
    }

    pub const fn rgb(r: f32, g: f32, b: f32) -> Self {
        Self::new(r, g, b, 1.0)
    }

    pub const fn grayscale(v: f32) -> Self {
        Self::new(v, v, v, 1.0)
    }

    pub fn into_raw(self) -> Vec4 {
        self.0
    }
}

impl From<Vec4> for Color {
    fn from(value: Vec4) -> Self {
        Self(value)
    }
}
