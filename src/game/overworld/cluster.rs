use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model::{PackedVertex, Vertex};
use crate::game::overworld::light_level::LightLevel;
use crate::game::overworld::occluder::Occluder;
use crate::game::registry::Registry;
use engine::renderer::vertex_mesh::VertexMeshCreate;
use engine::renderer::VertexMesh;
use engine::utils::{Int, MO_RELAXED};
use entity_data::{EntityBuilder, EntityId, EntityStorage};
use glm::{I32Vec3, U32Vec3, Vec3};
use lazy_static::lazy_static;
use nalgebra_glm as glm;
use nalgebra_glm::{I32Vec2, TVec3};
use parking_lot::{RwLock, RwLockReadGuard};
use std::convert::TryInto;
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

fn neighbour_index_from_pos(pos: &TVec3<usize>) -> usize {
    let p = pos.map(|v| (v > 0) as usize + (v == SIZE - 1) as usize);
    p.x * 9 + p.y * 3 + p.z
}

fn neighbour_index_from_dir(dir: &I32Vec3) -> usize {
    let p = dir.add_scalar(1);
    (p.x * 9 + p.y * 3 + p.z) as usize
}

pub fn neighbour_index_to_dir(index: usize) -> I32Vec3 {
    I32Vec3::new(index as i32 / 9 % 3, index as i32 / 3 % 3, index as i32 % 3).add_scalar(-1)
}

#[inline]
fn block_index(pos: &TVec3<usize>) -> usize {
    pos.x * SIZE * SIZE + pos.y * SIZE + pos.z
}

#[inline]
fn aligned_block_index(pos: &TVec3<usize>) -> usize {
    pos.x * ALIGNED_SIZE * ALIGNED_SIZE + pos.y * ALIGNED_SIZE + pos.z
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

    #[inline]
    fn calculate_lighting(&self, curr_block: IntrinsicBlockData) -> u16 {
        let corner = if self.corner.occluder.is_full() {
            curr_block.light_level
        } else {
            self.corner.light_level
        }
        .components();
        let side1 = if self.sides[0].occluder.is_full() {
            curr_block.light_level
        } else {
            self.sides[0].light_level
        }
        .components();
        let side2 = if self.sides[1].occluder.is_full() {
            curr_block.light_level
        } else {
            self.sides[1].light_level
        }
        .components();

        let color = (corner + side1 + side2) / 3;

        LightLevel::from_vec(color).bits()
    }
}

#[derive(Default, Debug, Clone, Copy)]
pub struct IntrinsicBlockData {
    pub occluder: Occluder,
    pub light_level: LightLevel,
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
    intrinsic_data: Vec<IntrinsicBlockData>,
    side_changed: [bool; 27],
    intrinsics_changed: AtomicBool,
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
            intrinsic_data: vec![Default::default(); ALIGNED_VOLUME],
            side_changed: [false; 27],
            intrinsics_changed: Default::default(),
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
    pub fn acquire_changed_sides(&mut self) -> [bool; 27] {
        let mut changed_sides = self.side_changed;
        changed_sides[1 * 9 + 1 * 3 + 1] = false;

        self.side_changed = [false; 27];
        changed_sides
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

        // TODO: set `side_changed` if necessary

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
        let pos: TVec3<usize> = glm::convert(*pos);

        let entity_id = &mut self.block_map[pos.x][pos.y][pos.z];
        if *entity_id != EntityId::NULL {
            self.block_storage.remove(entity_id);
        }
        let entity_builder = self.block_storage.add_entity(block.archetype() as u32);

        let occluder = if block.has_textured_model() {
            let model = self
                .registry
                .get_textured_block_model(block.textured_model())
                .unwrap();
            model.occluder()
        } else {
            Occluder::default()
        };

        self.side_changed[neighbour_index_from_pos(&pos)] = true;
        self.blocks[pos.x][pos.y][pos.z] = block;

        let index = aligned_block_index(&pos.add_scalar(1));
        self.intrinsic_data[index].occluder = occluder;

        BlockDataBuilder {
            entity_builder,
            entity_id,
        }
    }

    pub fn get_light_level(&mut self, pos: &U32Vec3) -> LightLevel {
        let index = aligned_block_index(&glm::convert(pos.add_scalar(1)));
        self.intrinsic_data[index].light_level
    }

    pub fn set_light_level(&mut self, pos: &U32Vec3, light_level: LightLevel) {
        let pos: TVec3<usize> = glm::convert(*pos);

        let index = aligned_block_index(&pos.add_scalar(1));
        self.intrinsic_data[index].light_level = light_level;

        self.side_changed[neighbour_index_from_pos(&pos)] = true;
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

        for i in 0..SIZE {
            for j in 0..SIZE {
                let p = map_pos(&dir, i, j, SIZE - 1).add_scalar(1);
                let index = aligned_block_index(&p);

                state &= self.intrinsic_data[index].occluder.occludes_side(facing);

                if !state {
                    break;
                }
            }
        }

        state
    }
    pub fn clear_outer_intrinsics(&mut self, side_offset: I32Vec3, value: IntrinsicBlockData) {
        let so = side_offset;
        let size = SIZE as i32;
        let b = so.map(|v| v == -size || v == size);

        if glm::all(&b) {
            self.clear_outer_intrinsics_corner(so, value);
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            self.clear_outer_intrinsics_edge(side_offset, value);
        } else {
            self.clear_outer_intrinsics_side(side_offset, value);
        }
    }

    fn clear_outer_intrinsics_side(&mut self, side_offset: I32Vec3, value: IntrinsicBlockData) {
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
        let dir = side_offset.map(|v| (v == (SIZE as i32)) as i32 - (v == -(SIZE as i32)) as i32);

        for k in 0..SIZE {
            for l in 0..SIZE {
                let dst_p = map_pos(&dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                let index = aligned_block_index(&dst_p);
                self.intrinsic_data[index] = value;
            }
        }

        self.intrinsics_changed.store(true, MO_RELAXED);
    }

    fn clear_outer_intrinsics_edge(&mut self, side_offset: I32Vec3, value: IntrinsicBlockData) {
        fn map_pos(m: &I32Vec3, k: usize, s: usize) -> TVec3<usize> {
            m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
        }

        let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= SIZE as i32) as i32);

        for k in 0..SIZE {
            let dst_p = map_pos(&m, k + 1, ALIGNED_SIZE - 1);
            let index = aligned_block_index(&dst_p);
            self.intrinsic_data[index] = value;
        }

        self.intrinsics_changed.store(true, MO_RELAXED);
    }

    fn clear_outer_intrinsics_corner(&mut self, side_offset: I32Vec3, value: IntrinsicBlockData) {
        let m = side_offset.map(|v| (v > 0) as u32);

        let dst_pos = m * (ALIGNED_SIZE - 1) as u32;
        let index = aligned_block_index(&glm::convert(dst_pos));
        self.intrinsic_data[index] = value;

        self.intrinsics_changed.store(true, MO_RELAXED);
    }

    pub fn paste_outer_intrinsics(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let so = side_offset;
        let size = SIZE as i32;
        let b = so.map(|v| v == -size || v == size);

        if glm::all(&b) {
            self.paste_outer_intrinsics_corner(side_cluster, so);
        } else if b.x && b.y || b.x && b.z || b.y && b.z {
            self.paste_outer_intrinsics_edge(side_cluster, side_offset);
        } else {
            self.paste_outer_intrinsics_side(side_cluster, side_offset);
        }
    }

    fn paste_outer_intrinsics_side(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
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
        let dir = side_offset.map(|v| (v == (SIZE as i32)) as i32 - (v == -(SIZE as i32)) as i32);
        let dir_inv = -dir;

        for k in 0..SIZE {
            for l in 0..SIZE {
                let dst_p = map_pos(&dir, k + 1, l + 1, ALIGNED_SIZE - 1);
                let src_p = map_pos(&dir_inv, k, l, SIZE - 1).add_scalar(1);
                let dst_index = aligned_block_index(&dst_p);
                let src_index = aligned_block_index(&src_p);

                self.intrinsic_data[dst_index] = side_cluster.intrinsic_data[src_index];
            }
        }

        self.intrinsics_changed.store(true, MO_RELAXED);
    }

    fn paste_outer_intrinsics_edge(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
        fn map_pos(m: &I32Vec3, k: usize, s: usize) -> TVec3<usize> {
            m.map(|v| k * (v == 0) as usize + s * (v > 0) as usize)
        }

        let m = side_offset.map(|v| -1 * (v < 0) as i32 + 1 * (v >= SIZE as i32) as i32);
        let n = -m;

        for k in 0..SIZE {
            let dst_p = map_pos(&m, k + 1, ALIGNED_SIZE - 1);
            let src_p = map_pos(&n, k, SIZE - 1).add_scalar(1);
            let dst_index = aligned_block_index(&dst_p);
            let src_index = aligned_block_index(&src_p);

            self.intrinsic_data[dst_index] = side_cluster.intrinsic_data[src_index];
        }

        self.intrinsics_changed.store(true, MO_RELAXED);
    }

    fn paste_outer_intrinsics_corner(&mut self, side_cluster: &Cluster, side_offset: I32Vec3) {
        let m = side_offset.map(|v| (v > 0) as u32);
        let n = side_offset.map(|v| (v < 0) as u32);

        let dst_p = m * (ALIGNED_SIZE - 1) as u32;
        let src_p = (n * (SIZE as u32 - 1)).add_scalar(1);
        let dst_index = aligned_block_index(&glm::convert(dst_p));
        let src_index = aligned_block_index(&glm::convert(src_p));

        self.intrinsic_data[dst_index] = side_cluster.intrinsic_data[src_index];

        self.intrinsics_changed.store(true, MO_RELAXED);
    }

    // TODO: Optimize this
    /// Returns a corner and two sides corresponding to the specified vertex on the given block and facing.
    fn get_vertex_neighbours(
        block_pos: &I32Vec3,
        vertex_pos: &Vec3,
        facing: Facing,
        intrinsic_data: &[IntrinsicBlockData],
    ) -> NeighbourVertexIntrinsics {
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

        let corner_index = aligned_block_index(&glm::try_convert(corner).unwrap());
        let side1_index = aligned_block_index(&glm::try_convert(side1).unwrap());
        let side2_index = aligned_block_index(&glm::try_convert(side2).unwrap());

        let corner = intrinsic_data[corner_index];
        let side1 = intrinsic_data[side1_index];
        let side2 = intrinsic_data[side2_index];

        NeighbourVertexIntrinsics {
            corner,
            sides: [side1, side2],
        }
    }

    pub fn update_mesh(&self) {
        fn add_vertices(out: &mut Vec<PackedVertex>, pos: Vec3, vertices: &[Vertex]) {
            out.extend(vertices.iter().cloned().map(|mut v| {
                v.position += pos;
                v.pack()
            }));
        }

        let intrinsics = &self.intrinsic_data;
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
                        let rel_index = aligned_block_index(&glm::try_convert(rel).unwrap());

                        let curr_intrinsics = intrinsics[rel_index];
                        let occludes = curr_intrinsics.occluder.occludes_side(facing.mirror());

                        if occludes {
                            continue;
                        }

                        for v in model.get_quads_by_facing(facing).chunks_exact(4) {
                            let mut v: [Vertex; 4] = v[0..4].try_into().unwrap();

                            v[0].position += posf;
                            v[1].position += posf;
                            v[2].position += posf;
                            v[3].position += posf;

                            let neighbours0 =
                                Self::get_vertex_neighbours(&pos, &v[0].position, facing, intrinsics);
                            let neighbours1 =
                                Self::get_vertex_neighbours(&pos, &v[1].position, facing, intrinsics);
                            let neighbours2 =
                                Self::get_vertex_neighbours(&pos, &v[2].position, facing, intrinsics);
                            let neighbours3 =
                                Self::get_vertex_neighbours(&pos, &v[3].position, facing, intrinsics);

                            // v[0].normal = glm::vec3(1.0, 1.0, 0.0);
                            // v[1].normal = glm::vec3(0.0, 1.0, 1.0);
                            // v[2].normal = glm::vec3(1.0, 0.0, 1.0);
                            // v[3].normal = glm::vec3(1.0, 0.0, 0.0);
                            v[0].ao = neighbours0.calculate_ao();
                            v[1].ao = neighbours1.calculate_ao();
                            v[2].ao = neighbours2.calculate_ao();
                            v[3].ao = neighbours3.calculate_ao();
                            v[0].lighting = neighbours0.calculate_lighting(curr_intrinsics);
                            v[1].lighting = neighbours1.calculate_lighting(curr_intrinsics);
                            v[2].lighting = neighbours2.calculate_lighting(curr_intrinsics);
                            v[3].lighting = neighbours3.calculate_lighting(curr_intrinsics);

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
