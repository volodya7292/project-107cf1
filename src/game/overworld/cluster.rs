use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model::{PackedVertex, Vertex};
use crate::game::registry::Registry;
use engine::renderer::vertex_mesh::VertexMeshCreate;
use engine::utils::MO_RELAXED;
use entity_data::{EntityBuilder, EntityId, EntityStorage, EntityStorageLayout};
use glm::{I32Vec3, U32Vec3, Vec3};
use nalgebra_glm as glm;
use nalgebra_glm::{I32Vec2, TVec3};

use engine::renderer::VertexMesh;
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard};
use std::convert::TryInto;
use std::ops::{BitAnd, BitAndAssign};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use vk_wrapper as vkw;

const SECTOR_SIZE: usize = 16;
const ALIGNED_SECTOR_SIZE: usize = SECTOR_SIZE + 2;
const SIZE_IN_SECTORS: usize = 4;
const VOLUME_IN_SECTORS: usize = SIZE_IN_SECTORS * SIZE_IN_SECTORS * SIZE_IN_SECTORS;
pub const SIZE: usize = SECTOR_SIZE * SIZE_IN_SECTORS;
pub const VOLUME: usize = SIZE * SIZE * SIZE;
const SECTOR_VOLUME: usize = SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE;
const ALIGNED_SECTOR_VOLUME: usize = ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE;

pub fn size(level: u32) -> u64 {
    SIZE as u64 * 2_u64.pow(level)
}

fn index_1d_to_3d(i: usize, ds: usize) -> [usize; 3] {
    let ds_sqr = ds * ds;
    let i_2d = i % ds_sqr;
    [i / ds_sqr, i_2d / ds, i_2d % ds]
}

fn sector_index(pos: U32Vec3) -> usize {
    SIZE_IN_SECTORS * SIZE_IN_SECTORS * pos.x as usize + SIZE_IN_SECTORS * pos.y as usize + pos.z as usize
}

const fn sector_pos(index: usize) -> U32Vec3 {
    U32Vec3::new(
        (index / (SIZE_IN_SECTORS * SIZE_IN_SECTORS)) as u32,
        (index % (SIZE_IN_SECTORS * SIZE_IN_SECTORS) / SIZE_IN_SECTORS) as u32,
        (index % SIZE_IN_SECTORS) as u32,
    )
}

fn block_pos_to_sector_index(cell_pos: &U32Vec3) -> usize {
    sector_index(cell_pos / (SECTOR_SIZE as u32))
}

#[derive(Default, Copy, Clone)]
pub struct Occluder(u8);

impl Occluder {
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

    pub const fn full() -> Occluder {
        Self::new(true, true, true, true, true, true)
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

#[derive(Default)]
struct VertexMeshCache {
    vertices: Vec<PackedVertex>,
    indices: Vec<u32>,
}

struct Sector {
    block_storage: EntityStorage,
    block_map: Box<[[[EntityId; SECTOR_SIZE]; SECTOR_SIZE]; SECTOR_SIZE]>,
    blocks: Box<[[[Block; SECTOR_SIZE]; SECTOR_SIZE]; SECTOR_SIZE]>,
    occluders: Mutex<Box<[[[Occluder; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]>>,
    // Non-atomic, allows fast mutation under `&mut self`
    changed: bool,
    // Atomic, allows mutation under `&self`
    changed2: AtomicBool,
    side_changed: [bool; 6],
    empty: AtomicBool,
    mesh_cache: Mutex<VertexMeshCache>,
}

impl Sector {
    fn new(layout: &EntityStorageLayout) -> Sector {
        Sector {
            block_storage: EntityStorage::new(layout),
            block_map: Default::default(),
            blocks: Default::default(),
            occluders: Default::default(),
            changed: false,
            changed2: AtomicBool::new(false),
            side_changed: [false; 6],
            empty: AtomicBool::new(true),
            mesh_cache: Default::default(),
        }
    }

    const fn entry_index(x: u32, y: u32, z: u32) -> usize {
        SECTOR_SIZE * SECTOR_SIZE * x as usize + SECTOR_SIZE * y as usize + z as usize
    }

    fn set_block(&mut self, pos: U32Vec3, block: Block) -> BlockDataBuilder {
        let pos: TVec3<usize> = glm::convert(pos);
        let entity_id = &mut self.block_map[pos.x][pos.y][pos.z];
        if *entity_id != EntityId::NULL {
            self.block_storage.remove(entity_id);
        }
        let entity_builder = self.block_storage.add_entity(block.archetype() as u32);

        let mut side_changed = self.side_changed;
        side_changed[0] |= pos.x == 0;
        side_changed[1] |= pos.x == SECTOR_SIZE - 1;
        side_changed[2] |= pos.y == 0;
        side_changed[3] |= pos.y == SECTOR_SIZE - 1;
        side_changed[4] |= pos.z == 0;
        side_changed[5] |= pos.z == SECTOR_SIZE - 1;

        self.blocks[pos.x][pos.y][pos.z] = block;
        self.side_changed = side_changed;
        self.changed = true;

        BlockDataBuilder {
            entity_builder,
            entity_id,
        }
    }

    fn check_edge_fully_occluded(&self, facing: Facing) -> bool {
        let dir = facing.direction();
        let mut state = true;

        fn map_pos(d: I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;
            let x = k * dy + k * dz + v * (d.x > 0) as usize;
            let y = k * dx + l * dz + v * (d.y > 0) as usize;
            let z = l * dx + l * dy + v * (d.z > 0) as usize;
            TVec3::new(x, y, z)
        }

        let occluders = self.occluders.lock();

        for i in 0..SECTOR_SIZE {
            for j in 0..SECTOR_SIZE {
                let p = map_pos(dir, i, j, SECTOR_SIZE - 1).add_scalar(1);
                state &= occluders[p[0]][p[1]][p[2]].occludes_side(facing);
                if !state {
                    break;
                }
            }
        }

        state
    }

    fn is_empty(&self) -> bool {
        self.empty.load(MO_RELAXED)
    }

    // TODO: optimize this
    fn calculate_ao(
        occluders: &[[[Occluder; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE],
        block_pos: I32Vec3,
        vertex_pos: Vec3,
        facing: Facing,
    ) -> f32 {
        let dir = facing.direction();

        let mut k: I32Vec2 = Default::default();
        let mut c = 0;
        let corner: I32Vec3 = glm::try_convert(vertex_pos.zip_zip_map(&dir, &block_pos, |v, d, b| {
            if d != 0 {
                if v.fract() > 0.0001 {
                    v.floor()
                } else {
                    if d < 0 {
                        v - 1.0
                    } else {
                        v
                    }
                }
            } else {
                k[c] = (v - b as f32 - 0.5).signum() as i32;
                c += 1;
                b as f32 + (v - b as f32 - 0.5).signum()
            }
        }))
        .unwrap();

        c = 0;
        let side1 = corner.zip_map(&dir, |v, d| {
            if d != 0 {
                v
            } else {
                c += 1;

                if c == 1 {
                    if k[0] < 0 && k[1] < 0 {
                        v
                    } else if k[0] < 0 && k[1] > 0 {
                        v + 1
                    } else if k[0] > 0 && k[1] > 0 {
                        v
                    } else {
                        v - 1
                    }
                } else {
                    if k[0] < 0 && k[1] < 0 {
                        v + 1
                    } else if k[0] < 0 && k[1] > 0 {
                        v
                    } else if k[0] > 0 && k[1] > 0 {
                        v - 1
                    } else {
                        v
                    }
                }
            }
        });

        c = 0;
        let side2 = corner.zip_map(&dir, |v, d| {
            if d != 0 {
                v
            } else {
                c += 1;

                if c == 1 {
                    if k[0] < 0 && k[1] < 0 {
                        v + 1
                    } else if k[0] < 0 && k[1] > 0 {
                        v
                    } else if k[0] > 0 && k[1] > 0 {
                        v - 1
                    } else {
                        v
                    }
                } else {
                    if k[0] < 0 && k[1] < 0 {
                        v
                    } else if k[0] < 0 && k[1] > 0 {
                        v - 1
                    } else if k[0] > 0 && k[1] > 0 {
                        v
                    } else {
                        v + 1
                    }
                }
            }
        });

        let corner = corner.add_scalar(1);
        let side1 = side1.add_scalar(1);
        let side2 = side2.add_scalar(1);

        let corner = occluders[corner.x as usize][corner.y as usize][corner.z as usize].0 != 0;
        let side1 = occluders[side1.x as usize][side1.y as usize][side1.z as usize].0 != 0;
        let side2 = occluders[side2.x as usize][side2.y as usize][side2.z as usize].0 != 0;

        !(side1 || side2 || corner) as u32 as f32
    }
}

pub struct BlockData<'a> {
    sector: &'a Sector,
    block: Block,
    id: EntityId,
}

pub trait BlockDataImpl {
    /// Returns specified block component
    fn get<C: 'static>(&self) -> Option<&C>;

    fn block(&self) -> Block;
}

impl BlockDataImpl for BlockData<'_> {
    fn get<C: 'static>(&self) -> Option<&C> {
        self.sector.block_storage.get::<C>(&self.id)
    }

    fn block(&self) -> Block {
        self.block
    }
}

pub struct BlockDataMut<'a> {
    sector: &'a mut Sector,
    block: Block,
    id: EntityId,
}

impl BlockDataImpl for BlockDataMut<'_> {
    fn get<C: 'static>(&self) -> Option<&C> {
        self.sector.block_storage.get::<C>(&self.id)
    }

    fn block(&self) -> Block {
        self.block
    }
}

impl BlockDataMut<'_> {
    pub fn get_mut<C: 'static>(&mut self) -> Option<&mut C> {
        self.sector.block_storage.get_mut::<C>(&self.id)
    }
}

pub struct BlockDataBuilder<'a> {
    entity_builder: EntityBuilder<'a>,
    entity_id: &'a mut EntityId,
}

impl BlockDataBuilder<'_> {
    pub fn with<C: 'static>(mut self, comp: C) -> Self {
        self.entity_builder = self.entity_builder.with::<C>(comp);
        self
    }

    pub fn build(self) {
        *self.entity_id = self.entity_builder.build();
    }
}

pub struct Cluster {
    registry: Arc<Registry>,
    sectors: [Sector; VOLUME_IN_SECTORS],
    side_changed: [bool; 6],
    device: Arc<vkw::Device>,
    vertex_mesh: Mutex<VertexMesh<PackedVertex, ()>>,
}

impl Cluster {
    /// Creating a new cluster is expensive due to its big size of memory
    pub fn new(registry: &Arc<Registry>, device: Arc<vkw::Device>) -> Self {
        let layout = registry.cluster_layout();
        let sectors: Vec<Sector> = (0..VOLUME_IN_SECTORS).map(|_| Sector::new(&layout)).collect();

        Self {
            registry: Arc::clone(registry),
            sectors: sectors.try_into().ok().unwrap(),
            side_changed: [false; 6],
            device,
            vertex_mesh: Default::default(),
        }
    }

    /// Returns a mask of `Facing` of changed cluster sides since the previous `Cluster::update_mesh()` call.
    pub fn changed_sides(&self) -> u8 {
        self.side_changed
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (i, b)| mask | ((*b as u8) << i))
    }

    /// Returns block data at `pos`
    pub fn get(&self, pos: &U32Vec3) -> BlockData {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::get_block failed: pos >= Cluster::SIZE");
        }
        if pos >= &U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let sector = self.sectors.get(block_pos_to_sector_index(pos)).unwrap();

        let s_pos = pos.map(|v| v % (SECTOR_SIZE as u32));
        let block = sector.blocks[s_pos.x as usize][s_pos.y as usize][s_pos.z as usize];
        let entity = sector.block_map[s_pos.x as usize][s_pos.y as usize][s_pos.z as usize];

        BlockData {
            sector,
            block,
            id: entity,
        }
    }

    /// Returns mutable block data at `pos`
    pub fn get_mut(&mut self, pos: &U32Vec3) -> BlockDataMut {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::get_block_mut failed: pos >= Cluster::SIZE");
        }
        if pos >= &U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let sector = self.sectors.get_mut(block_pos_to_sector_index(pos)).unwrap();

        let s_pos = pos.map(|v| v % (SECTOR_SIZE as u32));
        let block = sector.blocks[s_pos.x as usize][s_pos.y as usize][s_pos.z as usize];
        let entity = sector.block_map[s_pos.x as usize][s_pos.y as usize][s_pos.z as usize];

        BlockDataMut {
            sector,
            block,
            id: entity,
        }
    }

    /// Sets specified block at `pos`
    pub fn set(&mut self, pos: &U32Vec3, block: Block) -> BlockDataBuilder {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::set_block failed: pos >= Cluster::SIZE");
        }
        if pos >= &U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let sector = self.sectors.get_mut(block_pos_to_sector_index(pos)).unwrap();

        self.side_changed[0] |= pos.x == 0;
        self.side_changed[1] |= pos.x == (SIZE - 1) as u32;
        self.side_changed[2] |= pos.y == 0;
        self.side_changed[3] |= pos.y == (SIZE - 1) as u32;
        self.side_changed[4] |= pos.z == 0;
        self.side_changed[5] |= pos.z == (SIZE - 1) as u32;

        let s_pos = pos.map(|v| v % (SECTOR_SIZE as u32));
        sector.set_block(s_pos, block)
    }

    pub fn paste_outer_side_occlusion(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let so = side_offset;
        let size = SIZE as i32;
        let b = so.map(|v| v == -size || v == size);

        if glm::all(&b) {
            self.paste_outer_side_occlusion_corner(side_cluster, so);
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            self.paste_outer_side_occlusion_edge(side_cluster, side_offset);
        } else {
            self.paste_outer_side_occlusion_side(side_cluster, side_offset);
        }
    }

    /// Checks if self inner edge is fully occluded
    pub fn check_edge_fully_occluded(&self, facing: Facing) -> bool {
        let dir = facing.direction();
        let mut state = true;

        fn map_pos(d: I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;
            let x = k * dy + k * dz + v * (d.x > 0) as usize;
            let y = k * dx + l * dz + v * (d.y > 0) as usize;
            let z = l * dx + l * dy + v * (d.z > 0) as usize;
            TVec3::new(x, y, z)
        }

        for i in 0..SIZE_IN_SECTORS {
            for j in 0..SIZE_IN_SECTORS {
                let p: U32Vec3 = glm::convert(map_pos(dir, i, j, SIZE_IN_SECTORS - 1));
                state &= self.sectors[sector_index(p)].check_edge_fully_occluded(facing);
                if !state {
                    break;
                }
            }
        }

        state
    }

    fn paste_outer_side_occlusion_side(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
        fn map_sector_pos(d: I32Vec3, k: u32, l: u32) -> U32Vec3 {
            let dx = (d.x != 0) as u32;
            let dy = (d.y != 0) as u32;
            let dz = (d.z != 0) as u32;
            let x = k * dy + k * dz + (SIZE_IN_SECTORS as u32 - 1) * (d.x > 0) as u32;
            let y = k * dx + l * dz + (SIZE_IN_SECTORS as u32 - 1) * (d.y > 0) as u32;
            let z = l * dx + l * dy + (SIZE_IN_SECTORS as u32 - 1) * (d.z > 0) as u32;
            U32Vec3::new(x, y, z)
        }
        fn map_pos(d: I32Vec3, k: usize, l: usize, v: usize) -> TVec3<usize> {
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;
            let x = k * dy + k * dz + v * (d.x > 0) as usize;
            let y = k * dx + l * dz + v * (d.y > 0) as usize;
            let z = l * dx + l * dy + v * (d.z > 0) as usize;
            TVec3::new(x, y, z)
        }

        // Direction towards side cluster
        let dir = side_offset.map(|v| (v == (SIZE as i32)) as i32 - (v == -(SIZE as i32)) as i32);

        for i in 0..SIZE_IN_SECTORS {
            for j in 0..SIZE_IN_SECTORS {
                let dst_sector = &self.sectors[sector_index(map_sector_pos(dir, i as u32, j as u32))];
                let mut dst_occluders = dst_sector.occluders.lock();
                let src_sector =
                    &side_cluster.sectors[sector_index(map_sector_pos(-dir, i as u32, j as u32))];
                let src_occluders = src_sector.occluders.lock();

                for k in 0..SECTOR_SIZE {
                    for l in 0..SECTOR_SIZE {
                        let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SECTOR_SIZE - 1);
                        let src_p = map_pos(-dir, k, l, SECTOR_SIZE - 1).add_scalar(1);

                        dst_occluders[dst_p.x][dst_p.y][dst_p.z] = src_occluders[src_p.x][src_p.y][src_p.z];
                    }
                }

                dst_sector.changed2.store(true, MO_RELAXED);
            }
        }
    }

    fn paste_outer_side_occlusion_edge(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
        fn map_sector_pos(m: I32Vec3, i: u32) -> U32Vec3 {
            let s = SIZE_IN_SECTORS as u32 - 1;
            m.map(|v| i * (v == 0) as u32 + s * (v > 0) as u32)
        }
        fn map_pos(m: I32Vec3, k: usize, s: usize) -> TVec3<usize> {
            m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
        }

        let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= SIZE as i32) as i32);
        let n = -m;

        for i in 0..SIZE_IN_SECTORS {
            let dst_sector = &self.sectors[sector_index(map_sector_pos(m, i as u32))];
            let mut dst_occluders = dst_sector.occluders.lock();
            let src_sector = &side_cluster.sectors[sector_index(map_sector_pos(n, i as u32))];
            let src_occluders = src_sector.occluders.lock();

            for k in 0..SECTOR_SIZE {
                let dst_p = map_pos(m, k + 1, ALIGNED_SECTOR_SIZE - 1);
                let src_p = map_pos(n, k, SECTOR_SIZE - 1).add_scalar(1);

                dst_occluders[dst_p.x][dst_p.y][dst_p.z] = src_occluders[src_p.x][src_p.y][src_p.z];
            }

            dst_sector.changed2.store(true, MO_RELAXED);
        }
    }

    fn paste_outer_side_occlusion_corner(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let m = side_offset.map(|v| (v > 0) as u32);
        let n = side_offset.map(|v| (v < 0) as u32);

        let dst_p = m * (ALIGNED_SECTOR_SIZE - 1) as u32;
        let dst_sector_pos = m * (SIZE_IN_SECTORS - 1) as u32;
        let dst_sector = &self.sectors[sector_index(dst_sector_pos)];
        let mut dst_occluders = dst_sector.occluders.lock();

        let src_pos = (n * (SECTOR_SIZE - 1) as u32).add_scalar(1);
        let src_sector_pos = n * (SIZE_IN_SECTORS - 1) as u32;
        let src_sector = &side_cluster.sectors[sector_index(src_sector_pos)];
        let src_occluders = src_sector.occluders.lock();

        dst_occluders[dst_p.x as usize][dst_p.y as usize][dst_p.z as usize] =
            src_occluders[src_pos.x as usize][src_pos.y as usize][src_pos.z as usize];
        dst_sector.changed2.store(true, MO_RELAXED);
    }

    fn update_inner_side_occluders(&self) {
        macro_rules! side {
            ($sector_i: expr, $facing: expr, $k: ident, $l: ident, $x: expr, $y: expr, $z: expr, $x2: expr, $y2: expr, $z2: expr) => {
                let sector = &self.sectors[$sector_i];
                let occluders = sector.occluders.lock();
                let pos: I32Vec3 = glm::convert(sector_pos($sector_i));
                let j = $facing as usize;
                let rel = (pos + $facing.direction());

                if sector.side_changed[j]
                    && rel >= I32Vec3::from_element(0)
                    && rel < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                {
                    let side_sector = &self.sectors[sector_index(glm::try_convert(rel).unwrap())];
                    let mut side_occluders = side_sector.occluders.lock();
                    let facing_m = $facing.mirror();

                    for $k in 0..SECTOR_SIZE {
                        for $l in 0..SECTOR_SIZE {
                            side_occluders[$x][$y][$z]
                                .set_side($facing, occluders[$x2][$y2][$z2].occludes_side(facing_m));
                        }
                    }
                }
            };
        }
        macro_rules! edge {
            ($sector_i: expr, $side_offset: expr) => {
                let src_sector = &self.sectors[$sector_i];
                let src_occluders = src_sector.occluders.lock();
                let n = $side_offset;
                let m = -n;
                let src_sector_pos: I32Vec3 = glm::convert(sector_pos($sector_i));
                let dst_sector_pos = src_sector_pos + $side_offset;

                if dst_sector_pos >= I32Vec3::from_element(0)
                    && dst_sector_pos < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                {
                    let dst_sector = &self.sectors[sector_index(glm::try_convert(dst_sector_pos).unwrap())];
                    let mut dst_occluders = dst_sector.occluders.lock();

                    fn map_pos(m: I32Vec3, k: usize, s: usize) -> TVec3<usize> {
                        m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
                    }

                    for k in 0..SECTOR_SIZE {
                        let dst_p = map_pos(m, k + 1, ALIGNED_SECTOR_SIZE - 1);
                        let src_p = map_pos(n, k, SECTOR_SIZE - 1).add_scalar(1);

                        dst_occluders[dst_p.x][dst_p.y][dst_p.z] = src_occluders[src_p.x][src_p.y][src_p.z];
                    }
                }
            };
        }

        macro_rules! corner {
            ($sector_i: expr, $xyz: expr) => {
                let sector = &self.sectors[$sector_i];
                let occluders = sector.occluders.lock();

                if sector.changed {
                    let xyz = $xyz;
                    let xyz2 = xyz.map(|v| (ALIGNED_SECTOR_SIZE - 1) * (v == 1) as usize);
                    let pos: I32Vec3 = glm::convert(sector_pos($sector_i));
                    let rel = pos + glm::sign(&xyz.map(|v| v as i32 - SECTOR_SIZE as i32 / 2));

                    if rel >= I32Vec3::from_element(0) && rel < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                    {
                        let side_sector = &self.sectors[sector_index(glm::try_convert(rel).unwrap())];
                        let mut side_occluders = side_sector.occluders.lock();

                        side_occluders[xyz2.x][xyz2.y][xyz2.z] = occluders[xyz.x][xyz.y][xyz.z];
                    }
                }
            };
        }

        let m = ALIGNED_SECTOR_SIZE - 1;

        for i in 0..VOLUME_IN_SECTORS {
            side!(i, Facing::PositiveZ, k, l, k + 1, l + 1, 0, k + 1, l + 1, m - 1);
            side!(i, Facing::PositiveY, k, l, k + 1, 0, l + 1, k + 1, m - 1, l + 1);
            side!(i, Facing::PositiveX, k, l, 0, k + 1, l + 1, m - 1, k + 1, l + 1);
            side!(i, Facing::NegativeZ, k, l, k + 1, l + 1, m, k + 1, l + 1, 1);
            side!(i, Facing::NegativeY, k, l, k + 1, m, l + 1, k + 1, 1, l + 1);
            side!(i, Facing::NegativeX, k, l, m, k + 1, l + 1, 1, k + 1, l + 1);

            edge!(i, glm::vec3(0, -1, -1));
            edge!(i, glm::vec3(0, -1, 1));
            edge!(i, glm::vec3(0, 1, -1));
            edge!(i, glm::vec3(0, 1, 1));
            edge!(i, glm::vec3(-1, 0, -1));
            edge!(i, glm::vec3(-1, 0, 1));
            edge!(i, glm::vec3(1, 0, -1));
            edge!(i, glm::vec3(1, 0, 1));
            edge!(i, glm::vec3(-1, -1, 0));
            edge!(i, glm::vec3(-1, 1, 0));
            edge!(i, glm::vec3(1, -1, 0));
            edge!(i, glm::vec3(1, 1, 0));

            corner!(i, glm::vec3(1, 1, 1));
            corner!(i, glm::vec3(1, 1, SECTOR_SIZE));
            corner!(i, glm::vec3(1, SECTOR_SIZE, 1));
            corner!(i, glm::vec3(1, SECTOR_SIZE, SECTOR_SIZE));
            corner!(i, glm::vec3(SECTOR_SIZE, 1, 1));
            corner!(i, glm::vec3(SECTOR_SIZE, 1, SECTOR_SIZE));
            corner!(i, glm::vec3(SECTOR_SIZE, SECTOR_SIZE, 1));
            corner!(i, glm::vec3(SECTOR_SIZE, SECTOR_SIZE, SECTOR_SIZE));
        }
    }

    fn update_sector_blocks_occluders(&self, sector: &Sector) {
        let mut occluders = sector.occluders.lock();

        for x in 0..SECTOR_SIZE {
            for y in 0..SECTOR_SIZE {
                for z in 0..SECTOR_SIZE {
                    let block = &sector.blocks[x][y][z];

                    let occluder = if block.has_textured_model() {
                        let model = self
                            .registry
                            .get_textured_block_model(block.textured_model())
                            .unwrap();
                        model.occluder()
                    } else {
                        Occluder::default()
                    };

                    occluders[x + 1][y + 1][z + 1] = occluder;
                }
            }
        }
    }

    fn triangulate(&self, sector_index: usize) {
        let sector = &self.sectors[sector_index];
        let sector_pos: TVec3<f32> = glm::convert(sector_pos(sector_index) * (SECTOR_SIZE as u32));
        let occluders = sector.occluders.lock();

        self.update_sector_blocks_occluders(sector);

        let mut cache = &mut *sector.mesh_cache.lock();
        cache.vertices = Vec::<PackedVertex>::with_capacity(SECTOR_VOLUME * 8);

        fn add_vertices(out: &mut Vec<PackedVertex>, pos: Vec3, vertices: &[Vertex]) {
            out.extend(vertices.iter().cloned().map(|mut v| {
                v.position += pos;
                v.pack()
            }));
        }

        let mut empty = true;

        for x in 0..SECTOR_SIZE {
            for y in 0..SECTOR_SIZE {
                for z in 0..SECTOR_SIZE {
                    let pos = I32Vec3::new(x as i32, y as i32, z as i32);
                    let posf: Vec3 = glm::convert(pos);
                    let block = &sector.blocks[x][y][z];

                    if !block.has_textured_model() {
                        continue;
                    }
                    empty = false;

                    let model = self
                        .registry
                        .get_textured_block_model(block.textured_model())
                        .unwrap();

                    add_vertices(&mut cache.vertices, posf, model.get_inner_quads());

                    for i in 0..6 {
                        let facing = Facing::from_u8(i as u8);
                        let rel = (pos + facing.direction()).add_scalar(1);

                        let occludes = occluders[rel.x as usize][rel.y as usize][rel.z as usize]
                            .occludes_side(facing.mirror());

                        if !occludes {
                            for v in model.get_quads_by_facing(facing).chunks_exact(4) {
                                let mut v: [Vertex; 4] = v[0..4].try_into().unwrap();

                                v[0].position += posf;
                                v[1].position += posf;
                                v[2].position += posf;
                                v[3].position += posf;
                                // v[0].normal = glm::vec3(1.0, 1.0, 0.0);
                                // v[1].normal = glm::vec3(0.0, 1.0, 1.0);
                                // v[2].normal = glm::vec3(1.0, 0.0, 1.0);
                                // v[3].normal = glm::vec3(1.0, 0.0, 0.0);
                                v[0].ao = Sector::calculate_ao(&occluders, pos, v[0].position, facing);
                                v[1].ao = Sector::calculate_ao(&occluders, pos, v[1].position, facing);
                                v[2].ao = Sector::calculate_ao(&occluders, pos, v[2].position, facing);
                                v[3].ao = Sector::calculate_ao(&occluders, pos, v[3].position, facing);

                                if v[1].ao != v[2].ao {
                                    // if (1 - ao[0]) + (1 - ao[3]) > (1 - ao[1]) + (1 - ao[2]) {
                                    let vc = v;

                                    v[1] = vc[0];
                                    v[3] = vc[1];
                                    v[0] = vc[2];
                                    v[2] = vc[3];
                                }

                                for j in 0..4 {
                                    v[j].position += sector_pos;
                                }

                                cache.vertices.extend(v.map(|v| v.pack()));
                            }
                        }
                    }
                }
            }
        }

        sector.empty.store(empty, MO_RELAXED);

        cache.vertices.shrink_to_fit();

        let indices = &mut cache.indices;
        indices.resize(cache.vertices.len() / 4 * 6, 0);

        for i in (0..indices.len()).step_by(6) {
            let ind = (i / 6 * 4) as u32;
            indices[i] = ind;
            indices[i + 1] = ind + 2;
            indices[i + 2] = ind + 1;
            indices[i + 3] = ind + 2;
            indices[i + 4] = ind + 3;
            indices[i + 5] = ind + 1;
        }
    }

    /// Note: sets `Cluster::changed` to `false` and `Sector::changed` to `false` for all the sectors in the cluster.
    fn update_mesh(&self) {
        self.update_inner_side_occluders();

        for i in 0..VOLUME_IN_SECTORS {
            if self.sectors[i].changed || self.sectors[i].changed2.load(MO_RELAXED) {
                self.triangulate(i);
            }
        }

        let (v_count, i_count) = self.sectors.iter().fold((0, 0), |(v_count, i_count), sector| {
            let cache = sector.mesh_cache.lock();
            (v_count + cache.vertices.len(), i_count + cache.indices.len())
        });
        let mut vertices = Vec::<PackedVertex>::with_capacity(v_count);
        let mut indices = Vec::<u32>::with_capacity(i_count);

        for i in 0..VOLUME_IN_SECTORS {
            let sector = &self.sectors[i];
            let i_start = vertices.len() as u32;

            let cache = sector.mesh_cache.lock();
            let mut sector_indices = cache.indices.clone();
            for i in &mut sector_indices {
                *i += i_start;
            }

            vertices.extend(&cache.vertices);
            indices.extend(&sector_indices);
        }

        *self.vertex_mesh.lock() = self.device.create_vertex_mesh(&vertices, Some(&indices)).unwrap();
    }

    pub fn is_empty(&self) -> bool {
        self.sectors.iter().all(|sector| sector.is_empty())
    }
}

pub fn update_mesh(cluster: &RwLock<Cluster>) {
    let cluster = cluster.upgradable_read();

    cluster.update_mesh();

    // Remove `changed` markers
    let mut cluster = RwLockUpgradableReadGuard::upgrade(cluster);

    cluster.side_changed = [false; 6];

    for sector in &mut cluster.sectors {
        sector.changed = false;
        sector.side_changed = [false; 6];
        sector.changed2.store(false, MO_RELAXED);
    }
}
