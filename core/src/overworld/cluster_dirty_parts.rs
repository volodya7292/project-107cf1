use std::ops::{BitOr, BitOrAssign};

use nalgebra_glm as glm;

use crate::overworld::position::ClusterBlockPos;
use crate::overworld::raw_cluster;

#[derive(Copy, Clone, Default)]
pub struct ClusterDirtySides(u32);

impl ClusterDirtySides {
    const CENTER_PART_MASK: u32 = (raw_cluster::N_PARTS / 2) as u32;

    pub fn all() -> Self {
        Self((1 << raw_cluster::N_PARTS) - 1)
    }

    pub fn none() -> Self {
        Self::default()
    }

    pub fn set_dirty(&mut self, pos: &ClusterBlockPos) {
        let idx = raw_cluster::neighbour_index_from_pos(&glm::convert(*pos.get()));
        self.0 |= 1 << idx;
        // The center is never dirty
        self.0 &= !Self::CENTER_PART_MASK;
    }

    pub fn iter_sides(&self) -> impl Iterator<Item = usize> + '_ {
        (0..raw_cluster::N_PARTS).filter(|i| self.0 & (1 << i) != 0)
    }

    pub fn is_any(&self) -> bool {
        self.0 != 0
    }
}

impl BitOr for ClusterDirtySides {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ClusterDirtySides {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
