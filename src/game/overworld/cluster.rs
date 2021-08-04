use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model::{Quad, Vertex};
use crate::game::registry::Registry;
use crate::render_engine::material_pipeline::MaterialPipeline;
use crate::render_engine::vertex_mesh::VertexMeshCreate;
use crate::render_engine::{component, scene};
use crate::utils::{mesh_simplifier, HashMap, SliceSplitImpl};
use crate::{render_engine, utils};
use entity_data::{EntityBuilder, EntityId, EntityStorage, EntityStorageLayout};
use glm::{BVec3, I32Vec3, U32Vec3, Vec3};
use nalgebra_glm as glm;
use nalgebra_glm::{I32Vec2, TVec, TVec3, U32Vec2};
use rand_distr::num_traits::real::Real;
use smallvec::smallvec;
use std::collections::hash_map;
use std::convert::TryInto;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};
use std::sync::Arc;
use std::{iter, mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::PrimitiveTopology;

const SECTOR_SIZE: usize = 16;
const ALIGNED_SECTOR_SIZE: usize = SECTOR_SIZE + 2;
const SIZE_IN_SECTORS: usize = 4;
const VOLUME_IN_SECTORS: usize = SIZE_IN_SECTORS * SIZE_IN_SECTORS * SIZE_IN_SECTORS;
pub const SIZE: usize = SECTOR_SIZE * SIZE_IN_SECTORS;
pub const VOLUME: usize = SIZE * SIZE * SIZE;
const SECTOR_VOLUME: usize = SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE;
const ALIGNED_SECTOR_VOLUME: usize = ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE;

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

fn block_pos_to_sector_index(cell_pos: U32Vec3) -> usize {
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

struct Sector {
    block_storage: EntityStorage,
    block_map: Box<[[[EntityId; SECTOR_SIZE]; SECTOR_SIZE]; SECTOR_SIZE]>,
    blocks: Box<[[[Block; SECTOR_SIZE]; SECTOR_SIZE]; SECTOR_SIZE]>,
    occluders: Box<[[[Occluder; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]>,
    changed: bool,
    side_changed: [bool; 6],
    cache_vertices: Vec<Vertex>,
    cache_indices: Vec<u32>,
}

impl Sector {
    fn new(layout: &EntityStorageLayout) -> Sector {
        Sector {
            block_storage: EntityStorage::new(layout),
            block_map: Default::default(),
            blocks: Default::default(),
            occluders: Default::default(),
            changed: false,
            side_changed: [false; 6],
            cache_vertices: vec![],
            cache_indices: vec![],
        }
    }

    const fn entry_index(x: u32, y: u32, z: u32) -> usize {
        SECTOR_SIZE * SECTOR_SIZE * x as usize + SECTOR_SIZE * y as usize + z as usize
    }

    /// Returns whether the sector has been changed since the previous `Cluster::update_mesh()` call.
    pub fn changed(&self) -> bool {
        self.changed
    }

    fn set_block(&mut self, pos: U32Vec3, block: Block) -> BlockDataBuilder {
        let pos: TVec3<usize> = glm::convert(pos);
        let entity_id = &mut self.block_map[pos.x][pos.y][pos.z];
        if *entity_id != EntityId::NULL {
            self.block_storage.remove(entity_id);
        }
        let entity_builder = self.block_storage.add_entity(block.archetype());

        self.occluders[pos.x + 1][pos.y + 1][pos.z + 1] = Occluder::new(true, true, true, true, true, true); // TODO

        let mut side_changed = self.side_changed;
        side_changed[0] |= pos.x == 0;
        side_changed[1] |= pos.x == 15;
        side_changed[2] |= pos.y == 0;
        side_changed[3] |= pos.y == 15;
        side_changed[4] |= pos.z == 0;
        side_changed[5] |= pos.z == 15;

        self.blocks[pos.x][pos.y][pos.z] = block;
        self.side_changed = side_changed;
        self.changed = true;

        BlockDataBuilder {
            entity_builder,
            entity_id,
        }
    }

    // TODO: optimize this
    fn calculate_ao(&self, block_pos: I32Vec3, vertex_pos: Vec3, facing: Facing) -> f32 {
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

        // TODO: to account for boundaries, access global occluders instead of sector-local

        // TODO: try another method: use outer clusters to get global occluders on boundaries.
        // TODO: cache their locks to improve performance

        let corner = self.occluders[corner.x as usize][corner.y as usize][corner.z as usize].0 != 0;
        let side1 = self.occluders[side1.x as usize][side1.y as usize][side1.z as usize].0 != 0;
        let side2 = self.occluders[side2.x as usize][side2.y as usize][side2.z as usize].0 != 0;

        !(side1 || side2 || corner) as u32 as f32
    }
}

pub struct BlockData<'a> {
    sector: &'a Sector,
    id: EntityId,
}

impl BlockData<'_> {
    pub fn get<C: 'static>(&self) -> &C {
        todo!()
    }
}

pub struct BlockDataMut<'a> {
    sector: &'a mut Sector,
    id: EntityId,
}

impl BlockDataMut<'_> {
    pub fn get<C: 'static>(&self) -> &C {
        todo!()
    }

    pub fn get_mut<C: 'static>(&mut self) -> &mut C {
        todo!()
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
    entry_size: u32,
    changed: bool,
    device: Arc<vkw::Device>,
    vertex_mesh: render_engine::VertexMesh<Vertex, ()>,
}

impl Cluster {
    pub fn entry_size(&self) -> u32 {
        self.entry_size
    }

    /// Returns whether the cluster has been changed since the previous `Cluster::update_mesh()` call.
    pub fn changed(&self) -> bool {
        self.changed
    }

    pub fn vertex_mesh(&self) -> &render_engine::VertexMesh<Vertex, ()> {
        &self.vertex_mesh
    }

    pub fn get_block(&self, pos: U32Vec3) -> BlockData {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::get_block failed: pos >= Cluster::SIZE");
        }
        if pos >= U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let sector = self.sectors.get(block_pos_to_sector_index(pos)).unwrap();
        let entity = sector.block_map[pos.x as usize][pos.y as usize][pos.z as usize];

        BlockData { sector, id: entity }
    }

    pub fn get_block_mut(&mut self, pos: U32Vec3) -> BlockDataMut {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::get_block_mut failed: pos >= Cluster::SIZE");
        }
        if pos >= U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let sector = self.sectors.get_mut(block_pos_to_sector_index(pos)).unwrap();
        let entity = sector.block_map[pos.x as usize][pos.y as usize][pos.z as usize];

        BlockDataMut { sector, id: entity }
    }

    pub fn set_block(&mut self, pos: U32Vec3, block: Block) -> BlockDataBuilder {
        #[cold]
        #[inline(never)]
        fn assert_failed() -> ! {
            panic!("Cluster::set_block failed: pos >= Cluster::SIZE");
        }
        if pos >= U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let sector = self.sectors.get_mut(block_pos_to_sector_index(pos)).unwrap();
        let pos = pos.map(|v| v % (SECTOR_SIZE as u32));

        self.changed = true;
        // self.sides_changed |=
        //     glm::any(&pos.map(|v| v == 0)) || glm::any(&pos.map(|v| v == (SIZE as u32 - 1)));

        sector.set_block(pos, block)
    }

    pub fn clean_outer_side_occlusion(&mut self) {
        macro_rules! side_loop {
            ($i: ident, $j: ident, $k: ident, $l: ident, $cx: expr, $cy: expr, $cz: expr, $x: expr, $y: expr, $z: expr, $oi: expr) => {
                for $i in 0..SIZE_IN_SECTORS as u32 {
                    for $j in 0..SIZE_IN_SECTORS as u32 {
                        let sector = &mut self.sectors[sector_index(U32Vec3::new($cx, $cy, $cz))];
                        sector.changed = true;

                        for $k in 0..SECTOR_SIZE {
                            for $l in 0..SECTOR_SIZE {
                                sector.occluders[$x][$y][$z].clear_side($oi);
                            }
                        }
                    }
                }
            };
        }

        let n = (SIZE_IN_SECTORS - 1) as u32;
        let m = ALIGNED_SECTOR_SIZE - 1;

        side_loop!(i, j, k, l, i, j, 0, k + 1, l + 1, 0, Facing::PositiveZ);
        side_loop!(i, j, k, l, i, 0, j, k + 1, 0, l + 1, Facing::PositiveY);
        side_loop!(i, j, k, l, 0, i, j, 0, k + 1, l + 1, Facing::PositiveX);
        side_loop!(i, j, k, l, i, j, n, k + 1, l + 1, m, Facing::NegativeZ);
        side_loop!(i, j, k, l, i, n, j, k + 1, m, l + 1, Facing::NegativeY);
        side_loop!(i, j, k, l, n, i, j, m, k + 1, l + 1, Facing::NegativeX);
        self.changed = true;
    }

    pub fn paste_outer_side_occlusion(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let so = side_offset;
        let sc_size = SIZE as i32 * side_cluster.entry_size as i32;
        let self_size = SIZE as i32 * self.entry_size as i32;
        let b = so.map(|v| v == -sc_size || v == self_size);

        if glm::all(&b) {
            self.paste_outer_side_occlusion_corner(side_cluster, so);
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            self.paste_outer_side_occlusion_edge(side_cluster, side_offset);
        } else {
            self.paste_outer_side_occlusion_side(side_cluster, side_offset);
        }
    }

    fn paste_outer_side_occlusion_side(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
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

        let lb = -(SIZE as i32 * side_cluster.entry_size as i32);
        let rb = SIZE as i32 * self.entry_size as i32;

        // Direction towards side cluster
        let dir = side_offset.map(|v| (v == rb) as i32 - (v == lb) as i32);
        let facing = Facing::from_direction(dir);
        let facing_m = facing.mirror();

        if side_cluster.entry_size == self.entry_size {
            for i in 0..SIZE_IN_SECTORS {
                for j in 0..SIZE_IN_SECTORS {
                    let dst_sector = &mut self.sectors[sector_index(map_sector_pos(dir, i as u32, j as u32))];
                    let src_sector =
                        &side_cluster.sectors[sector_index(map_sector_pos(-dir, i as u32, j as u32))];

                    for k in 0..SECTOR_SIZE {
                        for l in 0..SECTOR_SIZE {
                            let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SECTOR_SIZE - 1);
                            let src_p = map_pos(-dir, k, l, SECTOR_SIZE - 1).add_scalar(1);

                            dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                                src_sector.occluders[src_p.x][src_p.y][src_p.z];
                        }
                    }

                    dst_sector.changed = true;
                }
            }
        } else if side_cluster.entry_size == (self.entry_size / 2) {
            let dabs = dir.abs();
            let is = (side_offset.x * dabs.z + side_offset.x * dabs.y + side_offset.y * dabs.x) as u32
                / self.entry_size
                / SECTOR_SIZE as u32;
            let js = (side_offset.y * dabs.z + side_offset.z * dabs.y + side_offset.z * dabs.x) as u32
                / self.entry_size
                / SECTOR_SIZE as u32;

            for i in 0..SIZE_IN_SECTORS {
                for j in 0..SIZE_IN_SECTORS {
                    let dst_sector = &mut self.sectors
                        [sector_index(map_sector_pos(dir, is + i as u32 / 2, js + j as u32 / 2))];
                    let src_sector =
                        &side_cluster.sectors[sector_index(map_sector_pos(-dir, i as u32, j as u32))];

                    let ks = (i % 2) * SECTOR_SIZE / 2;
                    let ls = (j % 2) * SECTOR_SIZE / 2;

                    for k in (0..SECTOR_SIZE).step_by(2) {
                        for l in (0..SECTOR_SIZE).step_by(2) {
                            let dst_p = map_pos(dir, ks + k / 2 + 1, ls + l / 2 + 1, ALIGNED_SECTOR_SIZE - 1);
                            let mut occluder = Occluder::default();
                            occluder.occlude_side(facing_m);

                            for k2 in 0..2 {
                                for l2 in 0..2 {
                                    let src_p = map_pos(-dir, k + k2, l + l2, SECTOR_SIZE - 1).add_scalar(1);
                                    occluder &= src_sector.occluders[src_p.x][src_p.y][src_p.z];
                                }
                            }

                            dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] = occluder;
                        }
                    }

                    dst_sector.changed = true;
                }
            }
        } else if side_cluster.entry_size == (self.entry_size * 2) {
            let dabs = dir.abs();
            let is = (side_offset.x * dabs.z + side_offset.x * dabs.y + side_offset.y * dabs.x).abs() as u32
                / side_cluster.entry_size
                / SECTOR_SIZE as u32;
            let js = (side_offset.y * dabs.z + side_offset.z * dabs.y + side_offset.z * dabs.x).abs() as u32
                / side_cluster.entry_size
                / SECTOR_SIZE as u32;

            for i in 0..SIZE_IN_SECTORS {
                for j in 0..SIZE_IN_SECTORS {
                    let dst_sector = &mut self.sectors[sector_index(map_sector_pos(dir, i as u32, j as u32))];
                    let src_sector = &side_cluster.sectors
                        [sector_index(map_sector_pos(-dir, is + i as u32 / 2, js + j as u32 / 2))];

                    let ks = (i % 2) * SECTOR_SIZE / 2;
                    let ls = (j % 2) * SECTOR_SIZE / 2;

                    // Note: 0..(SECTOR_SIZE + 1) : added 1 to compensate for lower-lod edges and corners.
                    for k in 0..(SECTOR_SIZE + 1) {
                        for l in 0..(SECTOR_SIZE + 1) {
                            let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SECTOR_SIZE - 1);
                            let src_p = map_pos(-dir, ks + k / 2, ls + l / 2, SECTOR_SIZE - 1).add_scalar(1);

                            dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                                src_sector.occluders[src_p.x][src_p.y][src_p.z];
                        }
                    }

                    dst_sector.changed = true;
                }
            }
        }

        self.changed = true;
    }

    fn paste_outer_side_occlusion_edge(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
        fn map_sector_pos(m: I32Vec3, i: u32) -> U32Vec3 {
            let s = SIZE_IN_SECTORS as u32 - 1;
            m.map(|v| i * (v == 0) as u32 + s * (v > 0) as u32)
        }
        fn map_pos(m: I32Vec3, k: usize, s: usize) -> TVec3<usize> {
            m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
        }

        let m =
            side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= SIZE as i32 * self.entry_size as i32) as i32);
        let n = -m;

        if side_cluster.entry_size == self.entry_size {
            for i in 0..SIZE_IN_SECTORS {
                let dst_sector = &mut self.sectors[sector_index(map_sector_pos(m, i as u32))];
                let src_sector = &side_cluster.sectors[sector_index(map_sector_pos(n, i as u32))];

                for k in 0..SECTOR_SIZE {
                    let dst_p = map_pos(m, k + 1, ALIGNED_SECTOR_SIZE - 1);
                    let src_p = map_pos(n, k, SECTOR_SIZE - 1).add_scalar(1);

                    dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                        src_sector.occluders[src_p.x][src_p.y][src_p.z];
                }

                dst_sector.changed = true;
            }
        } else if side_cluster.entry_size == (self.entry_size / 2) {
            let abs = m.map(|v| (v == 0) as i32);
            let is = (side_offset.x * abs.x + side_offset.y * abs.y + side_offset.z * abs.z)
                / self.entry_size as i32
                / SECTOR_SIZE as i32;

            for i in 0..SIZE_IN_SECTORS {
                let dst_sector = &mut self.sectors[sector_index(map_sector_pos(m, is as u32 + i as u32 / 2))];
                let src_sector = &side_cluster.sectors[sector_index(map_sector_pos(n, i as u32))];
                let ks = (i % 2) * SECTOR_SIZE / 2;

                for k in (0..SECTOR_SIZE).step_by(2) {
                    let dst_p = map_pos(m, ks + k / 2 + 1, ALIGNED_SECTOR_SIZE - 1);
                    let src_p = map_pos(n, k, SECTOR_SIZE - 2).add_scalar(1);
                    let mut occluder = Occluder::full();

                    for x in 0..2 {
                        for y in 0..2 {
                            for z in 0..2 {
                                let src_p = src_p + glm::vec3(x, y, z);
                                occluder &= src_sector.occluders[src_p.x][src_p.y][src_p.z];
                            }
                        }
                    }

                    dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] = occluder;
                }

                dst_sector.changed = true;
            }
        } else if side_cluster.entry_size == (self.entry_size * 2) {
            let abs = m.map(|v| (v == 0) as i32);
            let is = (side_offset.x * abs.x + side_offset.y * abs.y + side_offset.z * abs.z)
                / side_cluster.entry_size as i32
                / SECTOR_SIZE as i32;

            for i in 0..SIZE_IN_SECTORS {
                let dst_sector = &mut self.sectors[sector_index(map_sector_pos(m, i as u32))];
                let src_sector =
                    &side_cluster.sectors[sector_index(map_sector_pos(n, is as u32 + i as u32 / 2))];
                let ks = (i % 2) * SECTOR_SIZE / 2;

                for k in 0..SECTOR_SIZE {
                    let dst_p = map_pos(m, k, ALIGNED_SECTOR_SIZE - 1);
                    let src_p = map_pos(n, ks + k / 2, SECTOR_SIZE - 1).add_scalar(1);

                    dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                        src_sector.occluders[src_p.x][src_p.y][src_p.z];
                }

                dst_sector.changed = true;
            }
        }

        self.changed = true;
    }

    fn paste_outer_side_occlusion_corner(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let m = side_offset.map(|v| (v > 0) as u32);
        let n = side_offset.map(|v| (v < 0) as u32);

        let dst_p = m * (ALIGNED_SECTOR_SIZE - 1) as u32;
        let dst_sector_pos = m * (SIZE_IN_SECTORS - 1) as u32;
        let dst_sector = &mut self.sectors[sector_index(dst_sector_pos)];

        if side_cluster.entry_size == self.entry_size || side_cluster.entry_size == (self.entry_size * 2) {
            let src_pos = (n * (SECTOR_SIZE - 1) as u32).add_scalar(1);
            let src_sector_pos = n * (SIZE_IN_SECTORS - 1) as u32;
            let src_sector = &side_cluster.sectors[sector_index(src_sector_pos)];

            dst_sector.occluders[dst_p.x as usize][dst_p.y as usize][dst_p.z as usize] =
                src_sector.occluders[src_pos.x as usize][src_pos.y as usize][src_pos.z as usize];
            dst_sector.changed = true;
        } else if side_cluster.entry_size == (self.entry_size / 2) {
            let src_p = (n * (SECTOR_SIZE - 2) as u32).add_scalar(1);
            let src_sector_pos = n * (SIZE_IN_SECTORS - 1) as u32;
            let src_sector = &side_cluster.sectors[sector_index(src_sector_pos)];
            let mut occluder = Occluder::full();

            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        let src_p = src_p + U32Vec3::new(x, y, z);
                        occluder &=
                            src_sector.occluders[src_p.x as usize][src_p.y as usize][src_p.z as usize];
                    }
                }
            }

            dst_sector.occluders[dst_p.x as usize][dst_p.y as usize][dst_p.z as usize] = occluder;
            dst_sector.changed = true;
        }

        self.changed = true;
    }

    fn update_inner_side_occluders(&mut self) {
        macro_rules! side {
            ($sector_i: expr, $facing: expr, $k: ident, $l: ident, $x: expr, $y: expr, $z: expr, $x2: expr, $y2: expr, $z2: expr) => {
                let (sector, mut sectors) = self.sectors.split_mid_mut($sector_i).unwrap();
                let pos: I32Vec3 = glm::convert(sector_pos($sector_i));
                let j = $facing as usize;
                let rel = (pos + $facing.direction());

                if sector.side_changed[j]
                    && rel >= I32Vec3::from_element(0)
                    && rel < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                {
                    let side_sector = &mut sectors[sector_index(glm::try_convert(rel).unwrap())];
                    let facing_m = $facing.mirror();

                    for $k in 0..SECTOR_SIZE {
                        for $l in 0..SECTOR_SIZE {
                            side_sector.occluders[$x][$y][$z].set_side(
                                $facing,
                                sector.occluders[$x2][$y2][$z2].occludes_side(facing_m),
                            );
                        }
                    }

                    sector.side_changed[j] = false;
                }
            };
        }
        macro_rules! edge {
            ($sector_i: expr, $side_offset: expr) => {
                let (src_sector, mut sectors) = self.sectors.split_mid_mut($sector_i).unwrap();
                let n = $side_offset;
                let m = -n;
                let src_sector_pos: I32Vec3 = glm::convert(sector_pos($sector_i));
                let dst_sector_pos = src_sector_pos + $side_offset;

                if dst_sector_pos >= I32Vec3::from_element(0)
                    && dst_sector_pos < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                {
                    let dst_sector = &mut sectors[sector_index(glm::try_convert(dst_sector_pos).unwrap())];

                    fn map_pos(m: I32Vec3, k: usize, s: usize) -> TVec3<usize> {
                        m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
                    }

                    for k in 0..SECTOR_SIZE {
                        let dst_p = map_pos(m, k + 1, ALIGNED_SECTOR_SIZE - 1);
                        let src_p = map_pos(n, k, SECTOR_SIZE - 1).add_scalar(1);

                        dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                            src_sector.occluders[src_p.x][src_p.y][src_p.z];
                    }
                }
            };
        }

        macro_rules! corner {
            ($sector_i: expr, $xyz: expr) => {
                let (sector, mut sectors) = self.sectors.split_mid_mut($sector_i).unwrap();

                if sector.changed {
                    let xyz = $xyz;
                    let xyz2 = xyz.map(|v| (ALIGNED_SECTOR_SIZE - 1) * (v == 1) as usize);
                    let pos: I32Vec3 = glm::convert(sector_pos($sector_i));
                    let rel = pos + glm::sign(&xyz.map(|v| v as i32 - SECTOR_SIZE as i32 / 2));

                    if rel >= I32Vec3::from_element(0) && rel < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                    {
                        let side_sector = &mut sectors[sector_index(glm::try_convert(rel).unwrap())];
                        side_sector.occluders[xyz2.x][xyz2.y][xyz2.z] = sector.occluders[xyz.x][xyz.y][xyz.z];
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

    fn triangulate(&mut self, sector_index: usize) {
        let sector = &mut self.sectors[sector_index];
        let sector_pos: TVec3<f32> = glm::convert(sector_pos(sector_index) * (SECTOR_SIZE as u32));
        let scale = self.entry_size as f32;

        sector.cache_vertices = Vec::<Vertex>::with_capacity(SECTOR_VOLUME * 8);

        fn add_vertices(out: &mut Vec<Vertex>, pos: Vec3, scale: f32, vertices: &[Vertex]) {
            out.extend(vertices.iter().cloned().map(|mut v| {
                v.position += pos;
                v.position *= scale;
                v
            }));
        }

        for x in 0..SECTOR_SIZE {
            for y in 0..SECTOR_SIZE {
                for z in 0..SECTOR_SIZE {
                    let pos = I32Vec3::new(x as i32, y as i32, z as i32);
                    let posf: Vec3 = glm::convert(pos);
                    let block = &sector.blocks[x][y][z];

                    if block.is_none() {
                        continue;
                    }

                    let model = self
                        .registry
                        .get_textured_block_model(block.textured_model())
                        .unwrap();

                    add_vertices(&mut sector.cache_vertices, posf, scale, model.get_inner_quads());

                    for i in 0..6 {
                        let facing = Facing::from_u8(i as u8);
                        let rel = (pos + facing.direction()).add_scalar(1);

                        let occludes = sector.occluders[rel.x as usize][rel.y as usize][rel.z as usize]
                            .occludes_side(facing.mirror());

                        if !occludes {
                            for v in model.get_quads_by_facing(facing).chunks_exact(4) {
                                let mut v: [Vertex; 4] = v[0..4].try_into().unwrap();

                                v[0].position += posf;
                                v[1].position += posf;
                                v[2].position += posf;
                                v[3].position += posf;
                                v[0].normal = glm::vec3(1.0, 1.0, 0.0);
                                v[1].normal = glm::vec3(0.0, 1.0, 1.0);
                                v[2].normal = glm::vec3(1.0, 0.0, 1.0);
                                v[3].normal = glm::vec3(1.0, 0.0, 0.0);
                                v[0].ao = sector.calculate_ao(pos, v[0].position, facing);
                                v[1].ao = sector.calculate_ao(pos, v[1].position, facing);
                                v[2].ao = sector.calculate_ao(pos, v[2].position, facing);
                                v[3].ao = sector.calculate_ao(pos, v[3].position, facing);

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
                                    v[j].position *= scale;
                                }

                                sector.cache_vertices.extend(v);
                            }
                        }
                    }
                }
            }
        }

        sector.cache_vertices.shrink_to_fit();

        let mut indices = &mut sector.cache_indices;
        indices.resize(sector.cache_vertices.len() / 4 * 6, 0);

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
    pub fn update_mesh(&mut self) {
        if !self.changed {
            return;
        }
        self.changed = false;

        self.update_inner_side_occluders();

        for i in 0..VOLUME_IN_SECTORS {
            if self.sectors[i].changed {
                self.triangulate(i);
                self.sectors[i].changed = false;
            }
        }

        let (v_count, i_count) = self.sectors.iter().fold((0, 0), |(v_count, i_count), sector| {
            (
                v_count + sector.cache_vertices.len(),
                i_count + sector.cache_indices.len(),
            )
        });
        let mut vertices = Vec::<Vertex>::with_capacity(v_count);
        let mut indices = Vec::<u32>::with_capacity(i_count);

        for i in 0..VOLUME_IN_SECTORS {
            let sector = &self.sectors[i];
            let i_start = vertices.len() as u32;

            let mut sector_indices = sector.cache_indices.clone();
            for i in &mut sector_indices {
                *i += i_start;
            }

            vertices.extend(&sector.cache_vertices);
            indices.extend(&sector_indices);
        }

        self.vertex_mesh = self.device.create_vertex_mesh(&vertices, Some(&indices)).unwrap();
    }
}

pub fn new(registry: &Arc<Registry>, device: &Arc<vkw::Device>, node_size: u32) -> Cluster {
    let layout = registry.cluster_layout();
    let sectors: Vec<Sector> = (0..VOLUME_IN_SECTORS).map(|_| Sector::new(&layout)).collect();

    Cluster {
        registry: Arc::clone(registry),
        sectors: sectors.try_into().ok().unwrap(),
        entry_size: node_size,
        changed: false,
        device: Arc::clone(device),
        vertex_mesh: Default::default(),
    }
}
