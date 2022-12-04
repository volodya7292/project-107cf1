use std::ops::Deref;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct LiquidLevel(u8);

impl LiquidLevel {
    pub const MAX: Self = Self(15);
    pub const ZERO: Self = Self(0);

    pub const fn new(value: u8) -> Self {
        assert!(value <= Self::MAX.0);
        Self(value)
    }

    pub const fn get(&self) -> u8 {
        self.0
    }
}

impl Default for LiquidLevel {
    fn default() -> Self {
        Self::ZERO
    }
}
