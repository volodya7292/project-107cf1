use std::any::TypeId;
use std::collections::VecDeque;
use std::convert::TryInto;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use bit_vec::BitVec;
use entity_data::{ArchetypeState, Component, EntityId, EntityStorage};
use fixedbitset::FixedBitSet;
use glm::{I32Vec3, U32Vec3, Vec3};
use lazy_static::lazy_static;
use nalgebra_glm as glm;
use nalgebra_glm::{TVec3, U8Vec3, Vec2, Vec4};
use parking_lot::{RwLock, RwLockReadGuard};

use engine::renderer::vertex_mesh::VertexMeshCreate;
use engine::renderer::VertexMesh;
use engine::utils::MO_RELAXED;
use vk_wrapper as vkw;

use crate::game::overworld::block::{Block, BlockState};
use crate::game::overworld::block_component;
use crate::game::overworld::block_model::{quad_occludes_side, PackedVertex, Vertex};
use crate::game::overworld::cluster_dirty_parts::ClusterDirtySides;
use crate::game::overworld::facing::Facing;
use crate::game::overworld::light_level::LightLevel;
use crate::game::overworld::occluder::Occluder;
use crate::game::overworld::position::ClusterBlockPos;
use crate::game::registry::Registry;

pub const N_PARTS: usize = 27; // 1 (center) + 6 sides + 12 edges + 8 corners
const ALIGNED_SIZE: usize = RawCluster::SIZE + 2;
const ALIGNED_VOLUME: usize = ALIGNED_SIZE * ALIGNED_SIZE * ALIGNED_SIZE;

lazy_static! {
    static ref EMPTY_BLOCK_STORAGE: EntityStorage = EntityStorage::new();
}

pub fn neighbour_index_from_pos(pos: &TVec3<usize>) -> usize {
    let p = pos.map(|v| (v > 0) as usize + (v == RawCluster::SIZE - 1) as usize);
    p.x * 9 + p.y * 3 + p.z
}

fn neighbour_index_from_aligned_pos(pos: &TVec3<usize>) -> usize {
    let p = pos.map(|v| (v > 1) as usize + (v == RawCluster::SIZE) as usize);
    p.x * 9 + p.y * 3 + p.z
}

pub fn neighbour_index_from_dir(dir: &I32Vec3) -> usize {
    let p = dir.add_scalar(1);
    (p.x * 9 + p.y * 3 + p.z) as usize
}

pub fn neighbour_index_to_dir(index: usize) -> I32Vec3 {
    I32Vec3::new(index as i32 / 9 % 3, index as i32 / 3 % 3, index as i32 % 3).add_scalar(-1)
}

#[inline]
fn aligned_block_index(pos: &TVec3<usize>) -> usize {
    const ALIGNED_SIZE_SQR: usize = ALIGNED_SIZE * ALIGNED_SIZE;
    pos.x * ALIGNED_SIZE_SQR + pos.y * ALIGNED_SIZE + pos.z
}

struct NeighbourVertexIntrinsics {
    corner: IntrinsicBlockData,
    sides: [IntrinsicBlockData; 2],
}

impl NeighbourVertexIntrinsics {
    #[inline]
    fn calculate_ao(&self) -> u8 {
        (self.corner.occluder.is_empty()
            && self.sides[0].occluder.is_empty()
            && self.sides[1].occluder.is_empty()) as u8
            * 255
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IntrinsicBlockData {
    pub tex_model_id: u16,
    pub occluder: Occluder,
    pub light_level: LightLevel,
}

impl Default for IntrinsicBlockData {
    fn default() -> Self {
        Self {
            tex_model_id: u16::MAX,
            occluder: Default::default(),
            light_level: Default::default(),
        }
    }
}

pub struct BlockData<'a> {
    block_storage: &'a EntityStorage,
    inner_state: &'a InnerBlockState,
}

pub trait BlockDataImpl {
    fn block_id(&self) -> u16;
    /// Returns the specified block component `C`
    fn get<C: Component>(&self) -> Option<&C>;
}

impl BlockDataImpl for BlockData<'_> {
    fn block_id(&self) -> u16 {
        self.inner_state.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.inner_state.entity_id)
    }
}

pub struct BlockDataMut<'a> {
    block_storage: &'a mut EntityStorage,
    inner_state: &'a mut InnerBlockState,
}

impl BlockDataImpl for BlockDataMut<'_> {
    fn block_id(&self) -> u16 {
        self.inner_state.block_id
    }

    fn get<C: Component>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.inner_state.entity_id)
    }
}

impl BlockDataMut<'_> {
    pub fn get_mut<C: Component>(&mut self) -> Option<&mut C> {
        self.block_storage.get_mut::<C>(&self.inner_state.entity_id)
    }
}

#[derive(Copy, Clone)]
pub struct InnerBlockState {
    pub block_id: u16,
    pub entity_id: EntityId,
}

impl Default for InnerBlockState {
    fn default() -> Self {
        Self {
            block_id: u16::MAX,
            entity_id: Default::default(),
        }
    }
}

pub struct RawCluster {
    registry: Arc<Registry>,
    block_state_storage: EntityStorage,
    block_states: Vec<InnerBlockState>,
    intrinsic_data: Vec<IntrinsicBlockData>,
    empty: AtomicBool,
    vertex_mesh: RwLock<VertexMesh<PackedVertex, ()>>,
    vertex_mesh_translucent: RwLock<VertexMesh<PackedVertex, ()>>,
    light_addition_cache: VecDeque<TVec3<usize>>,
}

impl RawCluster {
    pub const SIZE: usize = 24;
    pub const VOLUME: usize = Self::SIZE * Self::SIZE * Self::SIZE;

    /// Creating a new cluster is expensive due to its big size of memory
    pub fn new(registry: &Arc<Registry>) -> Self {
        Self {
            registry: Arc::clone(registry),
            block_state_storage: EntityStorage::new(),
            block_states: vec![Default::default(); Self::VOLUME],
            intrinsic_data: vec![Default::default(); ALIGNED_VOLUME],
            empty: AtomicBool::new(true),
            vertex_mesh: Default::default(),
            vertex_mesh_translucent: Default::default(),
            light_addition_cache: VecDeque::with_capacity(Self::SIZE * Self::SIZE),
        }
    }

    pub fn vertex_mesh(&self) -> RwLockReadGuard<VertexMesh<PackedVertex, ()>> {
        self.vertex_mesh.read()
    }

    pub fn vertex_mesh_translucent(&self) -> RwLockReadGuard<VertexMesh<PackedVertex, ()>> {
        self.vertex_mesh_translucent.read()
    }

    /// Returns block data at `pos`
    pub fn get(&self, pos: &ClusterBlockPos) -> BlockData {
        BlockData {
            block_storage: &self.block_state_storage,
            inner_state: &self.block_states[pos.index()],
        }
    }

    /// Returns mutable block data at `pos`
    pub fn get_mut(&mut self, pos: &ClusterBlockPos) -> BlockDataMut {
        BlockDataMut {
            block_storage: &mut self.block_state_storage,
            inner_state: &mut self.block_states[pos.index()],
        }
    }

    /// Sets the specified block at `pos`
    pub fn set<A: ArchetypeState>(
        &mut self,
        pos: &ClusterBlockPos,
        block_state: BlockState<A>,
    ) -> BlockDataMut {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::set_block failed: pos >= Cluster::SIZE");
        }
        if pos.0 >= TVec3::from_element(Self::SIZE) {
            assert_failed();
        }

        let curr_state = &mut self.block_states[pos.index()];

        // Remove previous block state if present
        if curr_state.entity_id != EntityId::NULL {
            self.block_state_storage.remove(&curr_state.entity_id);
        }

        let entity_id = self.block_state_storage.add_entity(block_state.components);
        let block = self.registry.get_block(block_state.block_id).unwrap();

        // Update current state
        *curr_state = InnerBlockState {
            block_id: block_state.block_id,
            entity_id,
        };

        // Set intrinsics
        let index = aligned_block_index(&pos.0.add_scalar(1));
        self.intrinsic_data[index].occluder = block.occluder();
        self.intrinsic_data[index].tex_model_id = block.textured_model();

        BlockDataMut {
            block_storage: &mut self.block_state_storage,
            inner_state: curr_state,
        }
    }

    pub fn get_light_level(&self, pos: &ClusterBlockPos) -> LightLevel {
        let index = aligned_block_index(&pos.0.add_scalar(1));
        self.intrinsic_data[index].light_level
    }

    pub fn set_light_level(&mut self, pos: &ClusterBlockPos, light_level: LightLevel) {
        let index = aligned_block_index(&pos.0.add_scalar(1));
        self.intrinsic_data[index].light_level = light_level;
    }

    /// Checks if self inner edge is fully occluded
    pub fn check_edge_fully_occluded(&self, facing: Facing) -> bool {
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
                let p = map_pos(&dir, i, j, Self::SIZE - 1).add_scalar(1);
                let index = aligned_block_index(&glm::convert(p));

                state &= self.intrinsic_data[index].occluder.occludes_side(facing);

                if !state {
                    break;
                }
            }
        }

        state
    }

    pub fn clear_outer_intrinsics(&mut self, side_offset: I32Vec3, value: IntrinsicBlockData) {
        let size = Self::SIZE as i32;
        let b = side_offset.map(|v| v == -size || v == size);

        if glm::all(&b) {
            // Corner
            let m = side_offset.map(|v| (v > 0) as u32);
            let dst_pos = m * (ALIGNED_SIZE - 1) as u32;
            let index = aligned_block_index(&glm::convert(dst_pos));
            self.intrinsic_data[index] = value;
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            // Edge
            fn map_pos(m: &I32Vec3, k: usize, s: usize) -> TVec3<usize> {
                m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
            }

            let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= Self::SIZE as i32) as i32);

            for k in 0..Self::SIZE {
                let dst_p = map_pos(&m, k + 1, ALIGNED_SIZE - 1);
                let index = aligned_block_index(&glm::convert(dst_p));
                self.intrinsic_data[index] = value;
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
                    self.intrinsic_data[index] = value;
                }
            }
        }
    }

    pub fn paste_outer_intrinsics(&mut self, side_cluster: &RawCluster, side_offset: I32Vec3) {
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

            self.intrinsic_data[dst_index] = side_cluster.intrinsic_data[src_index];
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

                self.intrinsic_data[dst_index] = side_cluster.intrinsic_data[src_index];
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

                    self.intrinsic_data[dst_index] = side_cluster.intrinsic_data[src_index];
                }
            }
        }
    }

    pub fn propagate_outer_lighting(
        &mut self,
        side_cluster: &RawCluster,
        side_offset: I32Vec3,
    ) -> ClusterDirtySides {
        let size = Self::SIZE as i32;

        if side_offset.abs().sum() != size {
            // Not a side but corner or edge offset
            return ClusterDirtySides::none();
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

                self.intrinsic_data[dst_index].light_level =
                    side_cluster.intrinsic_data[src_index].light_level;
                self.light_addition_cache.push_back(glm::convert(dst_p));
            }
        }

        self.propagate_pending_lighting()
    }

    /// Propagates lighting from boundaries into the cluster
    fn propagate_pending_lighting(&mut self) -> ClusterDirtySides {
        const MAX_BOUNDARY: usize = (ALIGNED_SIZE - 1) as usize;

        let mut dirty_parts = ClusterDirtySides::none();

        while let Some(curr_pos) = self.light_addition_cache.pop_front() {
            let curr_index = aligned_block_index(&curr_pos);
            let curr_level = self.intrinsic_data[curr_index].light_level;
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

                let index = ClusterBlockPos(rel_pos - TVec3::from_element(1)).index();
                let aligned_index = aligned_block_index(&rel_pos);

                let block = self.block_states[index];
                let level = &mut self.intrinsic_data[aligned_index].light_level;
                let color = level.components();

                let block = self.registry.get_block(block.block_id).unwrap();

                if !self.registry.is_block_opaque(block)
                    && glm::any(&color.add_scalar(2).zip_map(&curr_color, |a, b| a <= b))
                {
                    let new_color = curr_color.map(|v| v.saturating_sub(1));
                    *level = LightLevel::from_vec(new_color);

                    self.light_addition_cache.push_back(glm::convert(rel_pos));

                    let dirty_pos = rel_pos.map(|v| v - 1);
                    dirty_parts.set_dirty(&ClusterBlockPos(dirty_pos));
                }
            }
        }

        dirty_parts
    }

    /// Propagates lighting where necessary in the cluster using light source at `block_pos`.
    pub fn propagate_lighting(&mut self, pos: &ClusterBlockPos) -> ClusterDirtySides {
        self.light_addition_cache
            .push_back(pos.offset(&TVec3::from_element(1)).0);
        self.propagate_pending_lighting()
    }

    /// Returns a corner and two sides corresponding to the specified vertex on the given block and facing.
    #[inline]
    fn get_vertex_neighbours(
        &self,
        block_pos: &I32Vec3,
        vertex_pos: &Vec3,
        facing: Facing,
    ) -> NeighbourVertexIntrinsics {
        let facing_dir = facing.direction();
        let facing_comp = facing_dir.iter().cloned().position(|v| v != 0).unwrap_or(0);

        let mut vertex_pos = *vertex_pos;
        vertex_pos[facing_comp] = 0.0;

        let mut center = Vec3::from_element(0.5);
        center[facing_comp] = 0.0;

        let side_dir: I32Vec3 = glm::convert_unchecked(glm::sign(&(vertex_pos - center)));

        let side1_comp = (facing_comp + 1) % 3;
        let side2_comp = (facing_comp + 2) % 3;

        let mut side1_dir = I32Vec3::default();
        side1_dir[side1_comp] = side_dir[side1_comp];

        let mut side2_dir = I32Vec3::default();
        side2_dir[side2_comp] = side_dir[side2_comp];

        let new_pos = block_pos.add_scalar(1) + facing_dir;
        let corner_pos = new_pos + side_dir;
        let side1_pos = new_pos + side1_dir;
        let side2_pos = new_pos + side2_dir;

        let corner_index = aligned_block_index(&glm::convert_unchecked(corner_pos));
        let side1_index = aligned_block_index(&glm::convert_unchecked(side1_pos));
        let side2_index = aligned_block_index(&glm::convert_unchecked(side2_pos));

        let corner = self.intrinsic_data[corner_index];
        let side1 = self.intrinsic_data[side1_index];
        let side2 = self.intrinsic_data[side2_index];

        NeighbourVertexIntrinsics {
            corner,
            sides: [side1, side2],
        }
    }

    #[inline]
    fn calc_quad_lighting(&self, block_pos: &I32Vec3, quad: &mut [Vertex; 4], facing: Facing) {
        const RELATIVE_SIDES: [[I32Vec3; 4]; 6] = [
            [
                Facing::NegativeY.direction(),
                Facing::NegativeZ.direction(),
                Facing::PositiveY.direction(),
                Facing::PositiveZ.direction(),
            ],
            [
                Facing::NegativeY.direction(),
                Facing::NegativeZ.direction(),
                Facing::PositiveY.direction(),
                Facing::PositiveZ.direction(),
            ],
            [
                Facing::NegativeX.direction(),
                Facing::NegativeZ.direction(),
                Facing::PositiveX.direction(),
                Facing::PositiveZ.direction(),
            ],
            [
                Facing::NegativeX.direction(),
                Facing::NegativeZ.direction(),
                Facing::PositiveX.direction(),
                Facing::PositiveZ.direction(),
            ],
            [
                Facing::NegativeX.direction(),
                Facing::NegativeY.direction(),
                Facing::PositiveX.direction(),
                Facing::PositiveY.direction(),
            ],
            [
                Facing::NegativeX.direction(),
                Facing::NegativeY.direction(),
                Facing::PositiveX.direction(),
                Facing::PositiveY.direction(),
            ],
        ];

        #[inline]
        fn calc_weights(weight_indices: &[usize; 8], v_comps: &[f32; 12]) -> Vec4 {
            Vec4::new(
                v_comps[weight_indices[0]] * v_comps[weight_indices[1]],
                v_comps[weight_indices[2]] * v_comps[weight_indices[3]],
                v_comps[weight_indices[4]] * v_comps[weight_indices[5]],
                v_comps[weight_indices[6]] * v_comps[weight_indices[7]],
            )
        }

        fn blend_lighting(base: LightLevel, mut neighbours: [LightLevel; 3]) -> Vec3 {
            for n in &mut neighbours {
                if n.is_zero() {
                    *n = base;
                }
            }

            glm::convert::<_, Vec3>(
                base.components()
                    + neighbours[0].components()
                    + neighbours[1].components()
                    + neighbours[2].components(),
            ) / (LightLevel::MAX_COMPONENT_VALUE as f32)
                * 0.25
        }

        let sides = &RELATIVE_SIDES[facing as usize];

        // FIXME: if block shape is not of full block, rel_pos = block_pos.
        let dir = facing.direction();
        let rel_pos = block_pos.add_scalar(1) + dir;

        let base = self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos))];
        let side0 = self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[0]))];
        let side1 = self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[1]))];
        let side2 = self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[2]))];
        let side3 = self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[3]))];

        let corner01 =
            self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[0] + sides[1]))];
        let corner12 =
            self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[1] + sides[2]))];
        let corner23 =
            self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[2] + sides[3]))];
        let corner30 =
            self.intrinsic_data[aligned_block_index(&glm::convert_unchecked(rel_pos + sides[3] + sides[0]))];

        // if min >= Vec3::from_element(1e-5) || max <= Vec3::from_element(1.0 - 1e-5) {
        // Do bilinear interpolation because the quad is not covering full block face.

        let lights = [
            blend_lighting(
                base.light_level,
                [side0.light_level, side1.light_level, corner01.light_level],
            ),
            blend_lighting(
                base.light_level,
                [side1.light_level, side2.light_level, corner12.light_level],
            ),
            blend_lighting(
                base.light_level,
                [side3.light_level, side0.light_level, corner30.light_level],
            ),
            blend_lighting(
                base.light_level,
                [side2.light_level, side3.light_level, corner23.light_level],
            ),
        ];

        let facing_comp = dir.iamax();
        let vi = (facing_comp + 1) % 3;
        let vj = (facing_comp + 2) % 3;

        for v in quad {
            let ij = Vec2::new(v.position[vi], v.position[vj]);
            let inv_v = Vec2::from_element(1.0) - ij;
            let weights = [inv_v.x * inv_v.y, ij.x * inv_v.y, inv_v.x * ij.y, ij.x * ij.y];

            let lighting = lights[0] * weights[0]
                + lights[1] * weights[1]
                + lights[2] * weights[2]
                + lights[3] * weights[3];

            v.lighting = LightLevel::from_color(lighting).bits();
        }

        // quad[vert_ids[0]].lighting = blend_lighting(
        //     base.light_level,
        //     [side0.light_level, side1.light_level, corner01.light_level],
        // )
        //     .bits();
        // quad[vert_ids[1]].lighting = blend_lighting(
        //     base.light_level,
        //     [side1.light_level, side2.light_level, corner12.light_level],
        // )
        //     .bits();
        // quad[vert_ids[2]].lighting = blend_lighting(
        //     base.light_level,
        //     [side2.light_level, side3.light_level, corner23.light_level],
        // )
        //     .bits();
        // quad[vert_ids[3]].lighting = blend_lighting(
        //     base.light_level,
        //     [side3.light_level, side0.light_level, corner30.light_level],
        // )
        //     .bits();

        // } else {
        // The quad covers full block face.
        // let lighting = neighbour_index_to_dir();
        // }
    }

    pub fn update_mesh(&self, device: &Arc<vkw::Device>) {
        #[inline]
        fn add_vertices(out: &mut Vec<PackedVertex>, pos: Vec3, vertices: &[Vertex]) {
            out.extend(vertices.iter().cloned().map(|mut v| {
                v.position += pos;
                v.pack()
            }));
        }

        let intrinsics = &self.intrinsic_data;
        let mut vertices = Vec::<PackedVertex>::with_capacity(Self::VOLUME * 8);
        let mut vertices_translucent = Vec::<PackedVertex>::with_capacity(Self::VOLUME * 8);
        let mut empty = true;

        for x in 0..Self::SIZE {
            for y in 0..Self::SIZE {
                for z in 0..Self::SIZE {
                    let pos = I32Vec3::new(x as i32, y as i32, z as i32);
                    let posf: Vec3 = glm::convert(pos);
                    let state = &self.block_states[ClusterBlockPos(glm::convert_unchecked(pos)).index()];
                    let block = self.registry.get_block(state.block_id).unwrap();

                    if !block.has_textured_model() {
                        continue;
                    }
                    empty = false;

                    let model = self
                        .registry
                        .get_textured_block_model(block.textured_model())
                        .unwrap();

                    if !model.get_inner_quads().is_empty() {
                        // TODO: REMOVE: For inner quads use the light level of the current block
                        // let aligned_pos = pos.add_scalar(1);
                        // let index = aligned_block_index(&glm::convert_unchecked(aligned_pos));
                        // let light_level = intrinsics[index].light_level;

                        for mut quad in model.get_inner_quads() {
                            let mut quad_vertices = quad.vertices;
                            let normal = engine::utils::calc_triangle_normal(
                                &quad_vertices[0].position,
                                &quad_vertices[1].position,
                                &quad_vertices[2].position,
                            );

                            self.calc_quad_lighting(
                                &pos,
                                &mut quad_vertices,
                                Facing::from_normal_closest(&normal),
                            );

                            let vertices_vec = if quad.transparent {
                                &mut vertices_translucent
                            } else {
                                &mut vertices
                            };

                            for mut v in quad_vertices {
                                v.position += posf;
                                v.normal = normal;

                                vertices_vec.push(v.pack());
                            }
                        }
                    }

                    let index = aligned_block_index(&glm::convert_unchecked(pos.add_scalar(1)));
                    let intrinsic_data = &intrinsics[index];

                    for i in 0..6 {
                        let facing = Facing::from_u8(i as u8);
                        let rel = (pos + facing.direction()).add_scalar(1);
                        let rel_index = aligned_block_index(&glm::convert_unchecked(rel));

                        let rel_intrinsic_data = intrinsics[rel_index];
                        let rel_occludes = rel_intrinsic_data.occluder.occludes_side(facing.mirror());

                        // Do not emit face if this side is fully occluded
                        if rel_occludes {
                            continue;
                        }

                        // Do not emit face if side faces are of the same shape
                        if model.merge_enabled()
                            && (intrinsic_data.tex_model_id == rel_intrinsic_data.tex_model_id
                                && model.side_shapes_equality()[facing.axis_idx()])
                        {
                            continue;
                        }
                        if let Some(rel_model) = self
                            .registry
                            .get_textured_block_model(rel_intrinsic_data.tex_model_id)
                        {
                            if model.merge_enabled()
                                && model
                                    .first_side_quad_vsorted(facing)
                                    .cmp_ordered(rel_model.first_side_quad_vsorted(facing.mirror()))
                            {
                                continue;
                            }
                        }

                        for quad in model.get_quads_by_facing(facing) {
                            let mut quad_vertices = quad.vertices;

                            let normal = engine::utils::calc_triangle_normal(
                                &quad_vertices[0].position,
                                &quad_vertices[1].position,
                                &quad_vertices[2].position,
                            );

                            self.calc_quad_lighting(&pos, &mut quad_vertices, facing);

                            for v in &mut quad_vertices {
                                let neighbours = self.get_vertex_neighbours(&pos, &v.position, facing);
                                v.position += posf;
                                v.normal = normal;
                                v.ao = neighbours.calculate_ao();
                                // v.lighting = neighbours.calculate_lighting(intrinsic);
                            }

                            if quad_vertices[1].ao != quad_vertices[2].ao
                                || quad_vertices[1].lighting != quad_vertices[2].lighting
                            {
                                let vc = quad_vertices;
                                quad_vertices[1] = vc[0];
                                quad_vertices[3] = vc[1];
                                quad_vertices[0] = vc[2];
                                quad_vertices[2] = vc[3];
                            }

                            let vertices_vec = if quad.transparent {
                                &mut vertices_translucent
                            } else {
                                &mut vertices
                            };

                            vertices_vec.extend(quad_vertices.map(|v| v.pack()));
                        }
                    }
                }
            }
        }

        self.empty.store(empty, MO_RELAXED);

        let mut indices = vec![0; vertices.len() / 4 * 6];
        let mut indices_translucent = vec![0; vertices_translucent.len() / 4 * 6];

        let map_quad_ids = |quad_idx, chunk: &mut [u32]| {
            let ind = (quad_idx * 4) as u32;
            chunk[0] = ind;
            chunk[1] = ind + 2;
            chunk[2] = ind + 1;
            chunk[3] = ind + 2;
            chunk[4] = ind + 3;
            chunk[5] = ind + 1;
        };

        for (i, chunk) in indices.chunks_exact_mut(6).enumerate() {
            map_quad_ids(i, chunk);
        }
        for (i, chunk) in indices_translucent.chunks_exact_mut(6).enumerate() {
            map_quad_ids(i, chunk);
        }

        *self.vertex_mesh.write() = device.create_vertex_mesh(&vertices, Some(&indices)).unwrap();
        *self.vertex_mesh_translucent.write() = device
            .create_vertex_mesh(&vertices_translucent, Some(&indices_translucent))
            .unwrap();
    }

    pub fn is_empty(&self) -> bool {
        self.empty.load(MO_RELAXED)
    }
}
