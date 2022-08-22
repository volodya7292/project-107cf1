use std::ops::{Range, RangeInclusive};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[repr(u8)]
pub enum BiomeSize {
    S64 = 6,
    M128 = 7,
    L256 = 8,
    XL512 = 9,
}

impl BiomeSize {
    pub const MIN: Self = BiomeSize::S64; // 2^6 = 64 blocks
    pub const MAX: Self = BiomeSize::XL512; // 2^9 = 512 blocks

    pub fn from_level(level: u8) -> Self {
        match level {
            6 => Self::S64,
            7 => Self::M128,
            8 => Self::L256,
            9 => Self::XL512,
            _ => panic!("Invalid biome level"),
        }
    }

    pub fn level(&self) -> u8 {
        *self as u8
    }
}

#[derive(Clone)]
pub struct Biome {
    /// Temperature in degrees Celsius
    temp_range: Range<f32>,
    /// 0.0 - dry, 1.0 - maximum moisture.
    moisture_range: Range<f32>,
    /// Altitude in blocks; 0 is the water level.
    altitude_range: Range<i32>,
    size_range: RangeInclusive<BiomeSize>,
    /// 0.0 - no generation, 1.0 - fully utilize the dedicated space for the biome.
    gen_probability: f32,
}

impl Biome {
    pub fn new(
        temp_range: Range<f32>,
        moisture_range: Range<f32>,
        altitude_range: Range<i32>,
        size_range: RangeInclusive<BiomeSize>,
        gen_probability: f32,
    ) -> Self {
        // Check absolute zero temperature boundary
        assert!(temp_range.start >= -273.15);

        Self {
            temp_range,
            moisture_range,
            altitude_range,
            size_range,
            gen_probability,
        }
    }
    pub fn size_range(&self) -> RangeInclusive<BiomeSize> {
        self.size_range.clone()
    }
}
