use std::ops::{BitOr, BitOrAssign};

use crate::overworld::position::{ClusterBlockPos, ClusterPos};
use crate::overworld::raw_cluster;
use crate::overworld::raw_cluster::RawCluster;

/// A set of of auxiliary cluster parts.
/// A cluster has 26 auxiliary parts: 6 sides, 12 edges, 8 corners.
#[derive(Copy, Clone)]
pub struct ClusterPartSet(u32);

impl ClusterPartSet {
    const CENTER_PART_MASK: u32 = 1 << (raw_cluster::N_PARTS / 2) as u32;
    pub const ALL: Self = Self(((1 << raw_cluster::N_PARTS) - 1) & !Self::CENTER_PART_MASK);
    pub const NONE: Self = Self(0);

    fn part_idx_from_cluster_block_pos(block_pos: &ClusterBlockPos) -> usize {
        let p = block_pos
            .get()
            .map(|v| (v > 0) as usize + (v == RawCluster::SIZE - 1) as usize);
        p.x * 9 + p.y * 3 + p.z
    }

    /// `this`: position of cluster which this `ClusterDirtySides` relates to;  
    /// `neighbour`: position of cluster which affects `this` cluster.
    #[inline]
    fn part_idx_from_relation(this: &ClusterPos, affecting_neighbour: &ClusterPos) -> usize {
        let diff = affecting_neighbour.get() - this.get();
        assert!(diff.amax() <= RawCluster::SIZE as i64);

        let diff_norm = (diff / RawCluster::SIZE as i64).add_scalar(1);
        (diff_norm.x * 9 + diff_norm.y * 3 + diff_norm.z) as usize
    }

    /// `pos`: inner block position from inside of cluster. x/y/z must be 0 or RawCluster::SIZE-1
    #[inline]
    pub fn set_from_block_pos(&mut self, pos: &ClusterBlockPos) {
        let idx = Self::part_idx_from_cluster_block_pos(pos);
        self.0 |= 1 << idx;
        self.0 &= !Self::CENTER_PART_MASK; // the center is never enabled
    }

    #[inline]
    pub fn set_from_idx(&mut self, part_idx: usize) {
        self.0 |= 1 << part_idx;
        self.0 &= !Self::CENTER_PART_MASK; // the center is never enabled
    }

    /// `this`: position of cluster which this `ClusterDirtySides` relates to;  
    /// `neighbour`: position of cluster which affects `this` cluster.
    #[inline]
    pub fn set_from_relation(&mut self, this: &ClusterPos, affecting_neighbour: &ClusterPos) {
        let idx = Self::part_idx_from_relation(this, affecting_neighbour);
        self.0 |= 1 << idx;
        self.0 &= !Self::CENTER_PART_MASK; // the center is never dirty
    }

    /// See [Self::set_from_relation].
    #[inline]
    pub fn clear_from_relation(&mut self, this: &ClusterPos, affecting_neighbour: &ClusterPos) {
        let idx = Self::part_idx_from_relation(this, affecting_neighbour);
        self.0 &= !(1 << idx);
        self.0 &= !Self::CENTER_PART_MASK; // the center is never dirty
    }

    #[inline]
    pub fn clear_from_idx(&mut self, part_idx: usize) {
        self.0 &= !(1 << part_idx);
    }

    /// Iterator over part indices.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        (0..raw_cluster::N_PARTS).filter(|i| self.0 & (1 << i) != 0)
    }

    pub fn has_any(&self) -> bool {
        self.0 != 0
    }
}

impl Default for ClusterPartSet {
    fn default() -> Self {
        Self::NONE
    }
}

impl BitOr for ClusterPartSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for ClusterPartSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}
