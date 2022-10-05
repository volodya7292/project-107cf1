use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, U32Vec3, U8Vec3};

use crate::game::overworld::raw_cluster;

/// Block position relative to cluster
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct ClusterBlockPos(pub U8Vec3);

impl ClusterBlockPos {
    #[inline]
    pub fn new(x: u8, y: u8, z: u8) -> Self {
        Self(glm::vec3(x, y, z))
    }

    #[inline]
    pub fn offset(&self, offset: &U8Vec3) -> Self {
        ClusterBlockPos(self.0 + offset)
    }

    #[inline]
    pub fn from_index(index: usize) -> Self {
        const SIZE_SQR: usize = raw_cluster::SIZE * raw_cluster::SIZE;

        let x = index / SIZE_SQR;
        let y = index % SIZE_SQR / raw_cluster::SIZE;
        let z = index % raw_cluster::SIZE;

        Self::new(x as u8, y as u8, z as u8)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0.x as usize * raw_cluster::SIZE * raw_cluster::SIZE
            + self.0.y as usize * raw_cluster::SIZE
            + self.0.z as usize
    }
}

/// Global block position
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct BlockPos(pub I64Vec3);

impl BlockPos {
    #[inline]
    pub fn new(x: i64, y: i64, z: i64) -> Self {
        Self(glm::vec3(x, y, z))
    }

    #[inline]
    pub fn from_f64(pos: &DVec3) -> Self {
        Self(glm::convert_unchecked(glm::floor(pos)))
    }

    #[inline]
    pub fn offset(&self, offset: &I64Vec3) -> Self {
        BlockPos(self.0 + offset)
    }

    #[inline]
    pub fn cluster_pos(&self) -> ClusterPos {
        ClusterPos(
            self.0
                .map(|v| v.div_euclid(raw_cluster::SIZE as i64) * raw_cluster::SIZE as i64),
        )
    }

    #[inline]
    pub fn cluster_block_pos(&self) -> ClusterBlockPos {
        ClusterBlockPos(self.0.map(|v| v.rem_euclid(raw_cluster::SIZE as i64) as u8))
    }
}

/// Global cluster position
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct ClusterPos(I64Vec3);

impl ClusterPos {
    #[inline]
    pub fn new(pos: I64Vec3) -> Self {
        debug_assert!(
            pos.x % raw_cluster::SIZE as i64 == 0
                && pos.y % raw_cluster::SIZE as i64 == 0
                && pos.z % raw_cluster::SIZE as i64 == 0
        );
        Self(pos)
    }

    #[inline]
    pub fn get(&self) -> &I64Vec3 {
        &self.0
    }

    #[inline]
    pub fn to_block_pos(self) -> BlockPos {
        BlockPos(self.0)
    }

    #[inline]
    pub fn offset(&self, offset: &I64Vec3) -> ClusterPos {
        ClusterPos(self.0 + offset * raw_cluster::SIZE as i64)
    }
}

impl From<ClusterPos> for I64Vec3 {
    fn from(pos: ClusterPos) -> Self {
        *pos.get()
    }
}
