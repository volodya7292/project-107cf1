use std::ops::{BitAnd, BitAndAssign};

use crate::game::overworld::facing::Facing;

#[derive(Copy, Clone, Default, Debug, Eq, PartialEq)]
pub struct Occluder(u8);

impl Occluder {
    /// Does not occlude any side
    pub const EMPTY: Self = Self::all(false);
    /// Occludes every side
    pub const FULL: Self = Self::all(true);

    pub const fn new(
        x_neg: bool,
        x_pos: bool,
        y_neg: bool,
        y_pos: bool,
        z_neg: bool,
        z_pos: bool,
    ) -> Occluder {
        Occluder(
            ((x_neg as u8) << (Facing::NegativeX as u8))
                | ((x_pos as u8) << (Facing::PositiveX as u8))
                | ((y_neg as u8) << (Facing::NegativeY as u8))
                | ((y_pos as u8) << (Facing::PositiveY as u8))
                | ((z_neg as u8) << (Facing::NegativeZ as u8))
                | ((z_pos as u8) << (Facing::PositiveZ as u8)),
        )
    }

    pub const fn all(occlude: bool) -> Self {
        Self::new(occlude, occlude, occlude, occlude, occlude, occlude)
    }

    pub const fn occludes_side(&self, facing: Facing) -> bool {
        ((self.0 >> (facing as u8)) & 1) == 1
    }

    pub fn occlude_side(&mut self, facing: Facing) {
        self.0 |= 1 << (facing as u8);
    }

    pub fn set_side(&mut self, facing: Facing, value: bool) {
        self.0 = (self.0 & !(1 << (facing as u8))) | ((value as u8) << (facing as u8));
    }

    pub fn clear_side(&mut self, facing: Facing) {
        self.0 &= !(1 << (facing as u8));
    }

    pub fn is_empty(&self) -> bool {
        self == &Self::EMPTY
    }

    pub fn is_full(&self) -> bool {
        self == &Self::FULL
    }
}

impl BitAnd for Occluder {
    type Output = Self;

    fn bitand(mut self, rhs: Self) -> Self::Output {
        self.0 &= rhs.0;
        self
    }
}

impl BitAndAssign for Occluder {
    fn bitand_assign(&mut self, rhs: Self) {
        *self = *self & rhs;
    }
}
