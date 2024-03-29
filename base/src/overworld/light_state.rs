use common::glm;
use glm::{U8Vec3, Vec3};
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Eq, PartialEq, Default, Debug, Serialize, Deserialize)]
pub struct LightLevel(u16);

impl LightLevel {
    pub const MAX_COMPONENT_VALUE: u8 = 31;
    pub const ZERO: Self = Self(0);
    pub const MAX: Self = Self::from_intensity(Self::MAX_COMPONENT_VALUE);

    pub fn is_zero(&self) -> bool {
        self == &Self::ZERO
    }

    /// Only 5 bits is available for each component
    pub const fn from_rgb(r: u8, g: u8, b: u8) -> Self {
        Self(((r as u16) << 10) | ((g as u16) << 5) | (b as u16))
    }

    pub const fn from_intensity(mut intensity: u8) -> Self {
        if intensity > Self::MAX_COMPONENT_VALUE {
            intensity = Self::MAX_COMPONENT_VALUE;
        }
        Self::from_rgb(intensity, intensity, intensity)
    }

    pub fn from_vec(components: U8Vec3) -> Self {
        Self::from_rgb(components.x, components.y, components.z)
    }

    pub fn from_color(color: Vec3) -> Self {
        let c = color * (Self::MAX_COMPONENT_VALUE as f32);
        Self(((c.x as u16) << 10) | ((c.y as u16) << 5) | (c.z as u16))
    }

    pub fn raw(&self) -> u16 {
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
        let f_vec: Vec3 = glm::convert(self.components());
        f_vec / (Self::MAX_COMPONENT_VALUE as f32)
    }
}
