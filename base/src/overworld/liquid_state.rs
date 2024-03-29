use serde::{Deserialize, Serialize};

#[derive(Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub struct LiquidState {
    liquid_id: u16,
    state: u8,
}

impl LiquidState {
    const SOURCE_MASK: u8 = 0b10000000;
    const LEVEL_MASK: u8 = 0b00001111;

    pub const NONE: Self = Self::new(u16::MAX, 0);
    pub const MAX_LEVEL: u8 = 15;

    pub const fn new(liquid_id: u16, value: u8) -> Self {
        assert!(value <= Self::MAX_LEVEL);
        Self {
            liquid_id,
            state: value,
        }
    }

    pub const fn source(liquid_id: u16) -> Self {
        Self {
            liquid_id,
            state: Self::MAX_LEVEL | Self::SOURCE_MASK,
        }
    }

    pub const fn max(liquid_id: u16) -> Self {
        Self {
            liquid_id,
            state: Self::MAX_LEVEL,
        }
    }

    pub const fn liquid_id(&self) -> u16 {
        self.liquid_id
    }

    pub const fn level(&self) -> u8 {
        self.state & Self::LEVEL_MASK
    }

    pub const fn is_empty(&self) -> bool {
        self.level() == 0
    }

    pub const fn is_max(&self) -> bool {
        self.level() == Self::MAX_LEVEL
    }

    pub const fn is_source(&self) -> bool {
        (self.state & Self::SOURCE_MASK) != 0
    }
}

impl Default for LiquidState {
    fn default() -> Self {
        Self::NONE
    }
}
