use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model::{PackedVertex, Vertex};
use crate::game::registry::Registry;
use engine::renderer::vertex_mesh::VertexMeshCreate;
use engine::renderer::VertexMesh;
use engine::utils::MO_RELAXED;
use entity_data::{EntityBuilder, EntityId, EntityStorage};
use glm::{I32Vec3, U32Vec3, Vec3};
use lazy_static::lazy_static;
use nalgebra_glm as glm;
use nalgebra_glm::{I32Vec2, TVec3};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard};
use std::convert::TryInto;
use std::ops::{BitAnd, BitAndAssign};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use vk_wrapper as vkw;

pub const SIZE: usize = 24;
pub const VOLUME: usize = SIZE * SIZE * SIZE;
const ALIGNED_SIZE: usize = SIZE + 2;
const ALIGNED_VOLUME: usize = ALIGNED_SIZE * ALIGNED_SIZE * ALIGNED_SIZE;

lazy_static! {
    static ref EMPTY_BLOCK_STORAGE: EntityStorage = EntityStorage::new(&Default::default());
}

pub fn size(level: u32) -> u64 {
    SIZE as u64 * 2_u64.pow(level)
}

fn index_1d_to_3d(i: usize, ds: usize) -> [usize; 3] {
    let ds_sqr = ds * ds;
    let i_2d = i % ds_sqr;
    [i / ds_sqr, i_2d / ds, i_2d % ds]
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

pub struct BlockData<'a> {
    block_storage: &'a EntityStorage,
    block: Block,
    id: EntityId,
}

impl BlockData<'_> {
    pub fn empty() -> Self {
        Self {
            block_storage: &EMPTY_BLOCK_STORAGE,
            block: Default::default(),
            id: Default::default(),
        }
    }
}

pub trait BlockDataImpl {
    /// Returns specified block component `C`
    fn get<C: 'static>(&self) -> Option<&C>;

    fn block(&self) -> Block;
}

impl BlockDataImpl for BlockData<'_> {
    fn get<C: 'static>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.id)
    }

    fn block(&self) -> Block {
        self.block
    }
}

pub struct BlockDataMut<'a> {
    block_storage: &'a mut EntityStorage,
    block: Block,
    id: EntityId,
}

impl BlockDataImpl for BlockDataMut<'_> {
    fn get<C: 'static>(&self) -> Option<&C> {
        self.block_storage.get::<C>(&self.id)
    }

    fn block(&self) -> Block {
        self.block
    }
}

impl BlockDataMut<'_> {
    pub fn get_mut<C: 'static>(&mut self) -> Option<&mut C> {
        self.block_storage.get_mut::<C>(&self.id)
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
    block_storage: EntityStorage,
    block_map: Box<[[[EntityId; SIZE]; SIZE]; SIZE]>,
    blocks: Box<[[[Block; SIZE]; SIZE]; SIZE]>,
    occluders: Mutex<Box<[[[Occluder; ALIGNED_SIZE]; ALIGNED_SIZE]; ALIGNED_SIZE]>>,
    side_changed: [bool; 6],
    occlusion_changed: AtomicBool,
    empty: AtomicBool,
    device: Arc<vkw::Device>,
    vertex_mesh: RwLock<VertexMesh<PackedVertex, ()>>,
}

impl Cluster {
    /// Creating a new cluster is expensive due to its big size of memory
    pub fn new(registry: &Arc<Registry>, device: Arc<vkw::Device>) -> Self {
        let layout = registry.cluster_layout();

        Self {
            registry: Arc::clone(registry),
            block_storage: EntityStorage::new(layout),
            block_map: Default::default(),
            blocks: Default::default(),
            occluders: Default::default(),
            side_changed: [false; 6],
            occlusion_changed: Default::default(),
            empty: AtomicBool::new(true),
            device,
            vertex_mesh: Default::default(),
        }
    }

    pub fn vertex_mesh(&self) -> RwLockReadGuard<VertexMesh<PackedVertex, ()>> {
        self.vertex_mesh.read()
    }

    /// Returns a mask of `Facing` of changed cluster sides since the previous `Cluster::update_mesh()` call,
    /// and clears it.
    pub fn acquire_changed_sides(&mut self) -> u8 {
        let mask = self
            .side_changed
            .iter()
            .enumerate()
            .fold(0_u8, |mask, (i, b)| mask | ((*b as u8) << i));
        self.side_changed = [false; 6];
        mask
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

        let pos: TVec3<usize> = glm::convert(*pos);
        let block = self.blocks[pos.x][pos.y][pos.z];
        let entity = self.block_map[pos.x][pos.y][pos.z];

        BlockData {
            block_storage: &self.block_storage,
            block,
            id: entity,
        }
    }

    /// Returns mutable block data at `pos`
    pub fn get_mut(&mut self, pos: &U32Vec3) -> BlockDataMut {
        #[cold]
        #[inline(never)]
        fn assert_failed() {
            panic!("Cluster::get_block_mut failed: pos >= Cluster::SIZE");
        }
        if pos >= &U32Vec3::from_element(SIZE as u32) {
            assert_failed();
        }

        let pos: TVec3<usize> = glm::convert(*pos);
        let block = self.blocks[pos.x][pos.y][pos.z];
        let entity = self.block_map[pos.x][pos.y][pos.z];

        BlockDataMut {
            block_storage: &mut self.block_storage,
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

        self.side_changed[0] |= pos.x == 0;
        self.side_changed[1] |= pos.x == (SIZE - 1) as u32;
        self.side_changed[2] |= pos.y == 0;
        self.side_changed[3] |= pos.y == (SIZE - 1) as u32;
        self.side_changed[4] |= pos.z == 0;
        self.side_changed[5] |= pos.z == (SIZE - 1) as u32;

        let pos: TVec3<usize> = glm::convert(*pos);
        let entity_id = &mut self.block_map[pos.x][pos.y][pos.z];
        if *entity_id != EntityId::NULL {
            self.block_storage.remove(entity_id);
        }
        let entity_builder = self.block_storage.add_entity(block.archetype() as u32);

        self.blocks[pos.x][pos.y][pos.z] = block;

        BlockDataBuilder {
            entity_builder,
            entity_id,
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

        let occluders = self.occluders.lock();

        for i in 0..SIZE {
            for j in 0..SIZE {
                let p = map_pos(dir, i, j, SIZE - 1).add_scalar(1);
                state &= occluders[p[0]][p[1]][p[2]].occludes_side(facing);
                if !state {
                    break;
                }
            }
        }

        state
    }
    pub fn clear_outer_side_occlusion(&self, side_offset: I32Vec3, value: Occluder) {
        let so = side_offset;
        let size = SIZE as i32;
        let b = so.map(|v| v == -size || v == size);

        if glm::all(&b) {
            self.clear_outer_side_occlusion_corner(so, value);
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            self.clear_outer_side_occlusion_edge(side_offset, value);
        } else {
            self.clear_outer_side_occlusion_side(side_offset, value);
        }
    }

    fn clear_outer_side_occlusion_side(&self, side_offset: I32Vec3, value: Occluder) {
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
        let mut dst_occluders = self.occluders.lock();

        for k in 0..SIZE {
            for l in 0..SIZE {
                let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                dst_occluders[dst_p.x][dst_p.y][dst_p.z] = value;
            }
        }

        self.occlusion_changed.store(true, MO_RELAXED);
    }

    fn clear_outer_side_occlusion_edge(&self, side_offset: I32Vec3, value: Occluder) {
        fn map_pos(m: I32Vec3, k: usize, s: usize) -> TVec3<usize> {
            m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
        }

        let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= SIZE as i32) as i32);
        let mut dst_occluders = self.occluders.lock();

        for k in 0..SIZE {
            let dst_p = map_pos(m, k + 1, ALIGNED_SIZE - 1);
            dst_occluders[dst_p.x][dst_p.y][dst_p.z] = value;
        }

        self.occlusion_changed.store(true, MO_RELAXED);
    }

    fn clear_outer_side_occlusion_corner(&self, side_offset: I32Vec3, value: Occluder) {
        let m = side_offset.map(|v| (v > 0) as u32);

        let dst_pos = m * (ALIGNED_SIZE - 1) as u32;
        let mut dst_occluders = self.occluders.lock();

        dst_occluders[dst_pos.x as usize][dst_pos.y as usize][dst_pos.z as usize] = value;

        self.occlusion_changed.store(true, MO_RELAXED);
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

    fn paste_outer_side_occlusion_side(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
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

        let mut dst_occluders = self.occluders.lock();
        let src_blocks = &side_cluster.blocks;

        for k in 0..SIZE {
            for l in 0..SIZE {
                let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                let src_blk_p = map_pos(-dir, k, l, SIZE - 1);

                let src_block = &src_blocks[src_blk_p.x][src_blk_p.y][src_blk_p.z];

                dst_occluders[dst_p.x][dst_p.y][dst_p.z] = side_cluster.block_occluder(src_block);
            }
        }

        self.occlusion_changed.store(true, MO_RELAXED);
    }

    fn paste_outer_side_occlusion_edge(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
        fn map_pos(m: I32Vec3, k: usize, s: usize) -> TVec3<usize> {
            m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
        }

        let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= SIZE as i32) as i32);
        let n = -m;

        let mut dst_occluders = self.occluders.lock();
        let src_blocks = &side_cluster.blocks;

        for k in 0..SIZE {
            let dst_p = map_pos(m, k + 1, ALIGNED_SIZE - 1);
            let src_blk_p = map_pos(n, k, SIZE - 1);

            let src_block = &src_blocks[src_blk_p.x][src_blk_p.y][src_blk_p.z];

            dst_occluders[dst_p.x][dst_p.y][dst_p.z] = side_cluster.block_occluder(src_block);
        }

        self.occlusion_changed.store(true, MO_RELAXED);
    }

    fn paste_outer_side_occlusion_corner(&self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let m = side_offset.map(|v| (v > 0) as u32);
        let n = side_offset.map(|v| (v < 0) as u32);

        let dst_pos = m * (ALIGNED_SIZE - 1) as u32;
        let mut dst_occluders = self.occluders.lock();

        let src_blk_p = glm::convert::<U32Vec3, TVec3<usize>>(n) * (SIZE - 1);
        let src_block = &self.blocks[src_blk_p.x][src_blk_p.y][src_blk_p.z];

        dst_occluders[dst_pos.x as usize][dst_pos.y as usize][dst_pos.z as usize] =
            side_cluster.block_occluder(src_block);

        self.occlusion_changed.store(true, MO_RELAXED);
    }

    fn block_occluder(&self, block: &Block) -> Occluder {
        if block.has_textured_model() {
            let model = self
                .registry
                .get_textured_block_model(block.textured_model())
                .unwrap();
            model.occluder()
        } else {
            Occluder::default()
        }
    }

    // TODO: optimize this
    fn calculate_ao(
        occluders: &[[[Occluder; ALIGNED_SIZE]; ALIGNED_SIZE]; ALIGNED_SIZE],
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

    fn update_mesh(&self) {
        let mut occluders = self.occluders.lock();

        // Update blocks occluders before calculating AO
        for x in 0..SIZE {
            for y in 0..SIZE {
                for z in 0..SIZE {
                    let block = &self.blocks[x][y][z];
                    occluders[x + 1][y + 1][z + 1] = self.block_occluder(block);
                }
            }
        }

        fn add_vertices(out: &mut Vec<PackedVertex>, pos: Vec3, vertices: &[Vertex]) {
            out.extend(vertices.iter().cloned().map(|mut v| {
                v.position += pos;
                v.pack()
            }));
        }

        let mut vertices = Vec::<PackedVertex>::with_capacity(VOLUME * 8);
        let mut empty = true;

        for x in 0..SIZE {
            for y in 0..SIZE {
                for z in 0..SIZE {
                    let pos = I32Vec3::new(x as i32, y as i32, z as i32);
                    let posf: Vec3 = glm::convert(pos);
                    let block = &self.blocks[x][y][z];

                    if !block.has_textured_model() {
                        continue;
                    }
                    empty = false;

                    let model = self
                        .registry
                        .get_textured_block_model(block.textured_model())
                        .unwrap();

                    add_vertices(&mut vertices, posf, model.get_inner_quads());

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
                                v[0].ao = Self::calculate_ao(&occluders, pos, v[0].position, facing);
                                v[1].ao = Self::calculate_ao(&occluders, pos, v[1].position, facing);
                                v[2].ao = Self::calculate_ao(&occluders, pos, v[2].position, facing);
                                v[3].ao = Self::calculate_ao(&occluders, pos, v[3].position, facing);

                                if v[1].ao != v[2].ao {
                                    // if (1 - ao[0]) + (1 - ao[3]) > (1 - ao[1]) + (1 - ao[2]) {
                                    let vc = v;

                                    v[1] = vc[0];
                                    v[3] = vc[1];
                                    v[0] = vc[2];
                                    v[2] = vc[3];
                                }

                                vertices.extend(v.map(|v| v.pack()));
                            }
                        }
                    }
                }
            }
        }

        self.empty.store(empty, MO_RELAXED);

        let mut indices = vec![0; vertices.len() / 4 * 6];

        for i in (0..indices.len()).step_by(6) {
            let ind = (i / 6 * 4) as u32;
            indices[i] = ind;
            indices[i + 1] = ind + 2;
            indices[i + 2] = ind + 1;
            indices[i + 3] = ind + 2;
            indices[i + 4] = ind + 3;
            indices[i + 5] = ind + 1;
        }

        *self.vertex_mesh.write() = self.device.create_vertex_mesh(&vertices, Some(&indices)).unwrap();
    }

    pub fn is_empty(&self) -> bool {
        self.empty.load(MO_RELAXED)
    }
}

pub fn update_mesh(cluster: &RwLock<Option<Cluster>>) {
    let cluster = cluster.upgradable_read();

    if cluster.is_none() {
        return;
    }

    cluster.as_ref().unwrap().update_mesh();

    // Remove `changed` markers
    let mut cluster = RwLockUpgradableReadGuard::upgrade(cluster);
    let mut cluster = cluster.as_mut().unwrap();

    cluster.side_changed = [false; 6];
}
