use crate::overworld::raw_cluster::RawCluster;
use common::glm;
use glm::{DVec3, I32Vec3, I64Vec3, TVec3};

/// Block position relative to cluster
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug)]
pub struct ClusterBlockPos(TVec3<usize>);

impl ClusterBlockPos {
    #[inline]
    pub const fn new(x: usize, y: usize, z: usize) -> Self {
        if x >= RawCluster::SIZE || y >= RawCluster::SIZE || z >= RawCluster::SIZE {
            panic!("Invalid coordinates");
        }
        Self(TVec3::new(x, y, z))
    }

    #[inline]
    pub const fn from_vec_unchecked(vec: TVec3<usize>) -> Self {
        Self(vec)
    }

    #[inline]
    pub fn get(&self) -> &TVec3<usize> {
        &self.0
    }

    #[inline]
    pub fn offset(&self, offset: &TVec3<usize>) -> Self {
        Self(self.0 + offset)
    }

    #[inline]
    pub fn from_index(index: usize) -> Self {
        const SIZE_SQR: usize = RawCluster::SIZE * RawCluster::SIZE;

        let x = index / SIZE_SQR;
        let y = index % SIZE_SQR / RawCluster::SIZE;
        let z = index % RawCluster::SIZE;

        Self::new(x, y, z)
    }

    #[inline]
    pub fn index(&self) -> usize {
        self.0.x * RawCluster::SIZE * RawCluster::SIZE + self.0.y * RawCluster::SIZE + self.0.z
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
        Self(self.0 + offset)
    }

    #[inline]
    pub fn offset_i32(&self, offset: &I32Vec3) -> Self {
        Self(self.0 + glm::convert::<_, I64Vec3>(*offset))
    }

    #[inline]
    pub fn cluster_pos(&self) -> ClusterPos {
        ClusterPos(
            self.0
                .map(|v| v.div_euclid(RawCluster::SIZE as i64) * RawCluster::SIZE as i64),
        )
    }

    #[inline]
    pub fn cluster_block_pos(&self) -> ClusterBlockPos {
        ClusterBlockPos(self.0.map(|v| v.rem_euclid(RawCluster::SIZE as i64) as usize))
    }
}

/// Global cluster position
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug)]
pub struct ClusterPos(I64Vec3);

impl ClusterPos {
    #[inline]
    pub fn new(pos: I64Vec3) -> Self {
        debug_assert!(
            pos.x % RawCluster::SIZE as i64 == 0
                && pos.y % RawCluster::SIZE as i64 == 0
                && pos.z % RawCluster::SIZE as i64 == 0
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
    pub fn offset(&self, offset: &I64Vec3) -> Self {
        Self(self.0 + offset * RawCluster::SIZE as i64)
    }

    #[inline]
    pub fn offset_i32(&self, offset: &I32Vec3) -> Self {
        Self(self.0 + glm::convert::<_, I64Vec3>(*offset) * RawCluster::SIZE as i64)
    }
}

impl From<ClusterPos> for I64Vec3 {
    fn from(pos: ClusterPos) -> Self {
        *pos.get()
    }
}

/// Relative block position in 3x3x3 cluster vicinity.
#[derive(Copy, Clone, Debug)]
pub struct RelativeBlockPos(pub I32Vec3);

impl RelativeBlockPos {
    #[inline]
    pub fn offset(&self, offset: &I32Vec3) -> Self {
        Self(self.0 + offset)
    }

    pub fn neighbour_pos(&self) -> TVec3<usize> {
        self.0
            .map(|v| (v >= 0) as usize + (v >= RawCluster::SIZE as i32) as usize)
    }

    pub fn cluster_idx(&self) -> usize {
        let p = self.neighbour_pos();
        p[0] * 9 + p[1] * 3 + p[2]
    }

    pub fn cluster_block_pos(&self) -> ClusterBlockPos {
        ClusterBlockPos(glm::convert_unchecked(
            self.0.map(|v| v.rem_euclid(RawCluster::SIZE as i32)),
        ))
    }
}

#[test]
fn relative_block_pos_works() {
    let pos = RelativeBlockPos(glm::vec3(-1, 0, 24));
    assert_eq!(pos.cluster_idx(), 5);
    assert_eq!(pos.cluster_block_pos(), ClusterBlockPos(glm::vec3(23, 0, 0)));
}
