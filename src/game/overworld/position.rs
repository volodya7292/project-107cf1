use nalgebra_glm as glm;
use nalgebra_glm::{I64Vec3, U32Vec3, U8Vec3};

use crate::game::overworld::raw_cluster;

/// Block position relative to cluster
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct ClusterBlockPos(pub U8Vec3);

impl ClusterBlockPos {}

/// Global block position
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct BlockPos(pub I64Vec3);

impl BlockPos {
    pub fn offset(&self, offset: &I64Vec3) -> BlockPos {
        BlockPos(self.0 + offset)
    }

    pub fn cluster_pos(&self) -> ClusterPos {
        ClusterPos(
            self.0
                .map(|v| v.div_euclid(raw_cluster::SIZE as i64) * raw_cluster::SIZE as i64),
        )
    }

    pub fn cluster_block_pos(&self) -> ClusterBlockPos {
        ClusterBlockPos(self.0.map(|v| v.rem_euclid(raw_cluster::SIZE as i64) as u8))
    }
}

/// Global cluster position
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
pub struct ClusterPos(pub I64Vec3);

impl ClusterPos {
    pub fn to_block_pos(self) -> BlockPos {
        BlockPos(self.0)
    }

    pub fn offset(&self, offset: &I64Vec3) -> ClusterPos {
        ClusterPos(self.0 + offset * raw_cluster::SIZE as i64)
    }
}
