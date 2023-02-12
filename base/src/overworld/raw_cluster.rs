use crate::overworld::block::BlockState;
use crate::overworld::cluster_part_set::ClusterPartSet;
use crate::overworld::facing::Facing;
use crate::overworld::light_state::LightState;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::occluder::Occluder;
use crate::overworld::position::ClusterBlockPos;
use crate::registry::Registry;
use common::glm;
use entity_data::{ArchetypeState, Component, EntityId, EntityStorage};
use glm::I32Vec3;
use glm::TVec3;
use std::collections::VecDeque;
use std::sync::Arc;

pub const N_PARTS: usize = 27; // 1 (center) + 6 sides + 12 edges + 8 corners

const ALIGNED_SIZE: usize = RawCluster::SIZE + 2;
const ALIGNED_VOLUME: usize = ALIGNED_SIZE * ALIGNED_SIZE * ALIGNED_SIZE;

// fn neighbour_index_from_aligned_pos(pos: &TVec3<usize>) -> usize {
//     let p = pos.map(|v| (v > 1) as usize + (v == RawCluster::SIZE) as usize);
//     p.x * 9 + p.y * 3 + p.z
// }

pub fn neighbour_index_from_dir(dir: &I32Vec3) -> usize {
    let p = dir.add_scalar(1);
    (p.x * 9 + p.y * 3 + p.z) as usize
}

pub fn neighbour_index_to_dir(index: usize) -> I32Vec3 {
    I32Vec3::new(index as i32 / 9 % 3, index as i32 / 3 % 3, index as i32 % 3).add_scalar(-1)
}

#[inline]
pub fn aligned_block_index(pos: &TVec3<usize>) -> usize {
    const ALIGNED_SIZE_SQR: usize = ALIGNED_SIZE * ALIGNED_SIZE;
    pos.x * ALIGNED_SIZE_SQR + pos.y * ALIGNED_SIZE + pos.z
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct CompactEntityId {
    arch_id: u16,
    /// Cluster size of 24^3 fits into 2^16 space.
    id: u16,
}

impl CompactEntityId {
    const NULL: Self = Self {
        arch_id: u16::MAX,
        id: u16::MAX,
    };

    fn new(entity_id: EntityId) -> Self {
        Self {
            arch_id: entity_id.archetype_id as u16,
            id: entity_id.id as u16,
        }
    }

    fn regular(&self) -> EntityId {
        EntityId::new(self.arch_id as u32, self.id as u32)
    }
}

impl Default for CompactEntityId {
    fn default() -> Self {
        Self::NULL
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CellInfo {
    pub entity_id: CompactEntityId,
    pub block_id: u16,
    pub occluder: Occluder,
    pub light_level: LightState,
    pub liquid_state: LiquidState,
    pub active: bool,
}

impl Default for CellInfo {
    fn default() -> Self {
        Self {
            entity_id: Default::default(),
            block_id: u16::MAX,
            occluder: Default::default(),
            light_level: Default::default(),
            liquid_state: LiquidState::NONE,
            active: false,
        }
    }
}

#[derive(Copy, Clone)]
pub struct BlockData<'a> {
    pub(super) block_storage: &'a EntityStorage,
    pub(super) info: &'a CellInfo,
}

pub trait BlockDataImpl {
    fn block_id(&self) -> u16;
    /// Returns the specified block component `C`
    fn get<C: Component>(&self) -> Option<&C>;

    fn light_state(&self) -> LightState;
    fn liquid_state(&self) -> &LiquidState;
    fn active(&self) -> bool;
}

impl BlockDataImpl for BlockData<'_> {
    fn block_id(&self) -> u16 {
        self.info.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.info.entity_id.regular())
    }

    fn light_state(&self) -> LightState {
        self.info.light_level
    }

    fn liquid_state(&self) -> &LiquidState {
        &self.info.liquid_state
    }

    fn active(&self) -> bool {
        self.info.active
    }
}

pub struct BlockDataMut<'a> {
    registry: &'a Registry,
    block_storage: &'a mut EntityStorage,
    info: &'a mut CellInfo,
}

impl BlockDataImpl for BlockDataMut<'_> {
    fn block_id(&self) -> u16 {
        self.info.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.info.entity_id.regular())
    }

    fn light_state(&self) -> LightState {
        self.info.light_level
    }

    fn liquid_state(&self) -> &LiquidState {
        &self.info.liquid_state
    }

    fn active(&self) -> bool {
        self.info.active
    }
}

impl BlockDataMut<'_> {
    pub fn set(&mut self, state: BlockState<impl ArchetypeState>) {
        let block = self.registry.get_block(state.block_id).unwrap();

        // Remove previous block state if present
        if self.info.entity_id != CompactEntityId::NULL {
            self.block_storage.remove(&self.info.entity_id.regular());
        }

        // Add new state to the storage
        let entity_id = self.block_storage.add(state.components);

        self.info.entity_id = CompactEntityId::new(entity_id);
        self.info.block_id = state.block_id;
        self.info.occluder = block.occluder();
        self.info.active |= block.active_by_default();
    }

    /// Raw mutable access to light level.
    pub fn light_state_mut(&mut self) -> &mut LightState {
        &mut self.info.light_level
    }

    /// Raw mutable access to liquid state.
    pub fn liquid_state_mut(&mut self) -> &mut LiquidState {
        &mut self.info.liquid_state
    }

    pub fn active_mut(&mut self) -> &mut bool {
        &mut self.info.active
    }

    pub fn get_mut<C: Component>(&mut self) -> Option<&mut C> {
        self.block_storage.get_mut::<C>(&self.info.entity_id.regular())
    }
}

pub struct RawCluster {
    registry: Arc<Registry>,
    block_state_storage: EntityStorage,
    cells: Vec<CellInfo>,
    light_addition_cache: VecDeque<TVec3<usize>>,
}

impl RawCluster {
    pub const SIZE: usize = 24;
    pub const VOLUME: usize = Self::SIZE * Self::SIZE * Self::SIZE;

    /// Creating a new cluster is expensive due to its big size of memory
    pub fn new(registry: &Arc<Registry>) -> Self {
        Self {
            registry: Arc::clone(registry),
            block_state_storage: Default::default(),
            cells: vec![Default::default(); ALIGNED_VOLUME],
            light_addition_cache: VecDeque::with_capacity(Self::SIZE * Self::SIZE),
        }
    }

    pub fn registry(&self) -> &Arc<Registry> {
        &self.registry
    }

    pub fn cells(&self) -> &[CellInfo] {
        &self.cells
    }

    /// Returns block data at `pos`.
    #[inline]
    pub fn get(&self, pos: &ClusterBlockPos) -> BlockData {
        let aligned_pos = pos.get().add_scalar(1);
        BlockData {
            block_storage: &self.block_state_storage,
            info: &self.cells[aligned_block_index(&aligned_pos)],
        }
    }

    /// Returns cell data at `pos`. Range: -1 <= pos[i] <= RawCluster::SIZE
    #[inline]
    pub fn get_cell(&self, pos: &I32Vec3) -> &CellInfo {
        let aligned_pos = pos.add_scalar(1);
        &self.cells[aligned_block_index(&glm::convert_unchecked(aligned_pos))]
    }

    /// Returns mutable block data at `pos`.
    #[inline]
    pub fn get_mut(&mut self, pos: &ClusterBlockPos) -> BlockDataMut {
        let aligned_pos = pos.get().add_scalar(1);
        BlockDataMut {
            registry: &self.registry,
            block_storage: &mut self.block_state_storage,
            info: &mut self.cells[aligned_block_index(&aligned_pos)],
        }
    }

    /// Checks if self inner edge is fully occluded
    pub fn is_side_fully_occluded(&self, facing: Facing) -> bool {
        let dir = facing.direction();
        let mut state = true;

        fn map_pos(d: &I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;
            let x = k * dy + k * dz + v * (d.x > 0) as usize;
            let y = k * dx + l * dz + v * (d.y > 0) as usize;
            let z = l * dx + l * dy + v * (d.z > 0) as usize;
            TVec3::new(x, y, z)
        }

        for i in 0..Self::SIZE {
            for j in 0..Self::SIZE {
                let p = map_pos(dir, i, j, Self::SIZE - 1).add_scalar(1);
                let index = aligned_block_index(&glm::convert(p));

                state &= self.cells[index].occluder.occludes_side(facing);

                if !state {
                    break;
                }
            }
        }

        state
    }

    pub fn clear_auxiliary_cells(&mut self, side_offset: I32Vec3, value: CellInfo) {
        let size = Self::SIZE as i32;
        let b = side_offset.map(|v| v == -size || v == size);

        if glm::all(&b) {
            // Corner
            let m = side_offset.map(|v| (v > 0) as u32);
            let dst_pos = m * (ALIGNED_SIZE - 1) as u32;
            let index = aligned_block_index(&glm::convert(dst_pos));
            self.cells[index] = value;
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            // Edge
            fn map_pos(m: &I32Vec3, k: usize, s: usize) -> TVec3<usize> {
                m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
            }

            let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= Self::SIZE as i32) as i32);

            for k in 0..Self::SIZE {
                let dst_p = map_pos(&m, k + 1, ALIGNED_SIZE - 1);
                let index = aligned_block_index(&glm::convert(dst_p));
                self.cells[index] = value;
            }
        } else {
            // Side
            fn map_pos(d: &I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
                let dx = (d.x != 0) as usize;
                let dy = (d.y != 0) as usize;
                let dz = (d.z != 0) as usize;
                let x = k * dy + k * dz + v * (d.x > 0) as usize;
                let y = k * dx + l * dz + v * (d.y > 0) as usize;
                let z = l * dx + l * dy + v * (d.z > 0) as usize;
                TVec3::new(x, y, z)
            }

            // Direction towards side cluster
            let dir =
                side_offset.map(|v| (v == (Self::SIZE as i32)) as i32 - (v == -(Self::SIZE as i32)) as i32);

            for k in 0..Self::SIZE {
                for l in 0..Self::SIZE {
                    let dst_p = map_pos(&dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                    let index = aligned_block_index(&glm::convert(dst_p));
                    self.cells[index] = value;
                }
            }
        }
    }

    pub fn paste_auxiliary_cells(&mut self, side_cluster: &RawCluster, side_offset: I32Vec3) {
        let size = Self::SIZE as i32;
        let b = side_offset.map(|v| v == -size || v == size);

        if glm::all(&b) {
            // Corner
            let m = side_offset.map(|v| (v > 0) as u32);
            let n = side_offset.map(|v| (v < 0) as u32);

            let dst_p = m * (ALIGNED_SIZE - 1) as u32;
            let src_p = (n * (Self::SIZE as u32 - 1)).add_scalar(1);
            let dst_index = aligned_block_index(&glm::convert(dst_p));
            let src_index = aligned_block_index(&glm::convert(src_p));

            self.cells[dst_index] = side_cluster.cells[src_index];
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            // Edge
            fn map_pos(m: &I32Vec3, k: usize, s: usize) -> TVec3<usize> {
                m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
            }

            let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= Self::SIZE as i32) as i32);
            let n = -m;

            for k in 0..Self::SIZE {
                let dst_p = map_pos(&m, k + 1, ALIGNED_SIZE - 1);
                let src_p = map_pos(&n, k, Self::SIZE - 1).add_scalar(1);
                let dst_index = aligned_block_index(&glm::convert(dst_p));
                let src_index = aligned_block_index(&glm::convert(src_p));

                self.cells[dst_index] = side_cluster.cells[src_index];
            }
        } else {
            // Side
            fn map_pos(d: &I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
                let dx = (d.x != 0) as usize;
                let dy = (d.y != 0) as usize;
                let dz = (d.z != 0) as usize;
                let x = k * dy + k * dz + v * (d.x > 0) as usize;
                let y = k * dx + l * dz + v * (d.y > 0) as usize;
                let z = l * dx + l * dy + v * (d.z > 0) as usize;
                TVec3::new(x, y, z)
            }

            // Direction towards side cluster
            let dir =
                side_offset.map(|v| (v == (Self::SIZE as i32)) as i32 - (v == -(Self::SIZE as i32)) as i32);
            let dir_inv = -dir;

            for k in 0..Self::SIZE {
                for l in 0..Self::SIZE {
                    let dst_p = map_pos(&dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                    let src_p = map_pos(&dir_inv, k, l, Self::SIZE - 1).add_scalar(1);
                    let dst_index = aligned_block_index(&glm::convert(dst_p));
                    let src_index = aligned_block_index(&glm::convert(src_p));

                    self.cells[dst_index] = side_cluster.cells[src_index];
                }
            }
        }
    }

    pub fn propagate_outer_lighting(
        &mut self,
        side_cluster: &RawCluster,
        side_offset: I32Vec3,
    ) -> ClusterPartSet {
        let size = Self::SIZE as i32;

        if side_offset.abs().sum() != size {
            // Not a side but corner or edge offset
            return ClusterPartSet::NONE;
        }

        fn map_pos(d: &I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;
            let x = k * dy + k * dz + v * (d.x > 0) as usize;
            let y = k * dx + l * dz + v * (d.y > 0) as usize;
            let z = l * dx + l * dy + v * (d.z > 0) as usize;
            TVec3::new(x, y, z)
        }

        // Direction towards side cluster
        let dir = side_offset.map(|v| (v == (Self::SIZE as i32)) as i32 - (v == -(Self::SIZE as i32)) as i32);
        let dir_inv = -dir;

        for k in 0..Self::SIZE {
            for l in 0..Self::SIZE {
                let dst_p = map_pos(&dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                let src_p = map_pos(&dir_inv, k, l, Self::SIZE - 1).add_scalar(1);
                let dst_index = aligned_block_index(&glm::convert(dst_p));
                let src_index = aligned_block_index(&glm::convert(src_p));

                self.cells[dst_index].light_level = side_cluster.cells[src_index].light_level;
                self.light_addition_cache.push_back(glm::convert(dst_p));
            }
        }

        self.propagate_pending_lighting()
    }

    /// Propagates lighting from boundaries into the cluster
    fn propagate_pending_lighting(&mut self) -> ClusterPartSet {
        const MAX_BOUNDARY: usize = (ALIGNED_SIZE - 1) as usize;

        let mut dirty_parts = ClusterPartSet::NONE;

        while let Some(curr_pos) = self.light_addition_cache.pop_front() {
            let curr_index = aligned_block_index(&curr_pos);
            let curr_level = self.cells[curr_index].light_level;
            let curr_color = curr_level.components();

            for i in 0..6 {
                let dir = Facing::DIRECTIONS[i];
                let rel_pos =
                    glm::convert_unchecked::<_, TVec3<usize>>(glm::convert::<_, I32Vec3>(curr_pos) - dir);

                if rel_pos.x < 1
                    || rel_pos.y < 1
                    || rel_pos.z < 1
                    || rel_pos.x >= MAX_BOUNDARY
                    || rel_pos.y >= MAX_BOUNDARY
                    || rel_pos.z >= MAX_BOUNDARY
                {
                    continue;
                }

                let aligned_index = aligned_block_index(&rel_pos);
                let rel_cell = &mut self.cells[aligned_index];

                let rel_block = self.registry.get_block(rel_cell.block_id).unwrap();
                let rel_level = &mut rel_cell.light_level;
                let rel_color = rel_level.components();

                if rel_block.is_opaque() {
                    continue;
                }
                if glm::all(&rel_color.add_scalar(2).zip_map(&curr_color, |a, b| a > b)) {
                    continue;
                }

                let new_color = curr_color.map(|v| v.saturating_sub(1));
                *rel_level = LightState::from_vec(new_color);

                let dirty_pos = rel_pos.map(|v| v - 1);
                dirty_parts.set_from_block_pos(&ClusterBlockPos::from_vec_unchecked(dirty_pos));

                self.light_addition_cache.push_back(glm::convert(rel_pos));
            }
        }

        dirty_parts
    }

    /// Propagates lighting where necessary in the cluster using light source at `block_pos`.
    pub fn propagate_lighting(&mut self, pos: &ClusterBlockPos) -> ClusterPartSet {
        self.light_addition_cache.push_back(pos.get().add_scalar(1));
        self.propagate_pending_lighting()
    }
}
