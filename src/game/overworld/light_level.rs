use nalgebra_glm::{U8Vec3, Vec3};
use std::ops::Add;

#[derive(Copy, Clone, Default, Debug)]
pub struct LightLevel(u16);

impl LightLevel {
    /// Only 5 bit is available for each component
    pub fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self(((r as u16) << 10) | ((g as u16) << 5) | (b as u16))
    }

    pub fn from_intensity(intensity: u8) -> Self {
        Self::from_rgb(intensity, intensity, intensity)
    }

    pub fn from_color(color: Vec3) -> Self {
        let c = color * 255.0;
        Self(((c.x as u16) << 10) | ((c.y as u16) << 5) | (c.z as u16))
    }

    pub fn bits(&self) -> u16 {
        self.0
    }

    pub fn components(&self) -> U8Vec3 {
        U8Vec3::new(
            ((self.0 >> 10) & 0x1f) as u8,
            ((self.0 >> 5) & 0x1f) as u8,
            (self.0 & 0x1f) as u8,
        )
    }

    pub fn color(&self) -> Vec3 {
        Vec3::new(
            ((self.0 >> 10) & 0x1f) as f32,
            ((self.0 >> 5) & 0x1f) as f32,
            (self.0 & 0x1f) as f32,
        ) / 255.0
    }
}
