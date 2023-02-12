use std::ops::RangeInclusive;

/// Degrees Celsius
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MeanTemperature {
    TNeg30 = -30,
    TNeg22 = -22,
    TNeg15 = -15,
    TNeg7 = -7,
    T0 = 0,
    TPos7 = 7,
    TPos15 = 12,
    TPos22 = 22,
    TPos30 = 30,
}

impl MeanTemperature {
    pub const MIN: Self = Self::TNeg30;
    pub const MAX: Self = Self::TPos30;
    pub const SPREAD: f32 = (Self::MAX as i32 - Self::MIN as i32) as f32;
}

/// Relative humidity, percentage
#[derive(Copy, Clone, Eq, PartialEq)]
pub enum MeanHumidity {
    H0 = 0,
    H12 = 12,
    H25 = 25,
    H37 = 37,
    H50 = 50,
    H62 = 62,
    H75 = 75,
    H87 = 87,
    H100 = 100,
}

impl MeanHumidity {
    pub const MIN: Self = Self::H0;
    pub const MAX: Self = Self::H100;
    pub const SPREAD: f32 = (Self::MAX as i32 - Self::MIN as i32) as f32;
}

#[derive(Clone)]
pub struct Biome {
    temp_range: RangeInclusive<MeanTemperature>,
    humidity_range: RangeInclusive<MeanHumidity>,
    /// Range: [-1, 1]; 0 is the surface level
    altitude_range: RangeInclusive<f32>,
}

impl Biome {
    pub fn new(
        temp_range: RangeInclusive<MeanTemperature>,
        humidity_range: RangeInclusive<MeanHumidity>,
        altitude_range: RangeInclusive<f32>,
    ) -> Self {
        assert!(*temp_range.start() as i32 <= *temp_range.end() as i32);
        assert!(*humidity_range.start() as u32 <= *humidity_range.end() as u32);
        assert!(*altitude_range.start() <= *altitude_range.end());
        assert!(*altitude_range.start() >= -1.0 && *altitude_range.end() <= 1.0);

        if (*temp_range.end() as i32 - *temp_range.start() as i32).abs() < 15 {
            panic!("Min temperature spread is 15 degrees!");
        }
        if (*humidity_range.end() as u32 - *humidity_range.start() as u32) < 25 {
            panic!("Min humidity spread is 25%!");
        }

        Self {
            temp_range,
            humidity_range,
            altitude_range,
        }
    }

    pub fn temp_range(&self) -> RangeInclusive<MeanTemperature> {
        self.temp_range.clone()
    }

    pub fn humidity_range(&self) -> RangeInclusive<MeanHumidity> {
        self.humidity_range.clone()
    }

    pub fn altitude_range(&self) -> RangeInclusive<f32> {
        self.altitude_range.clone()
    }
}
