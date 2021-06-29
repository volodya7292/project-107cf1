use std::collections::hash_map;
use std::convert::TryInto;
use std::sync::Arc;
use std::{iter, mem, slice};

use entity_data::{EntityBuilder, EntityId, EntityStorage, EntityStorageLayout};
use glm::{BVec3, I32Vec3, U32Vec3, Vec3};
use nalgebra_glm as glm;
use smallvec::smallvec;

use vk_wrapper as vkw;
use vk_wrapper::PrimitiveTopology;

use crate::game::overworld::block::BlockProps;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model::{Quad, Vertex};
use crate::game::registry::GameRegistry;
use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::vertex_mesh::VertexMeshCreate;
use crate::renderer::{component, scene};
use crate::utils::{mesh_simplifier, HashMap, SliceSplitImpl};
use crate::{renderer, utils};
use nalgebra::Vector3;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

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

// pub struct Entry {
//     content_id: u16,
//     secondary_content_id: u16,
//     data_id: u16,
//     orientation: [u8; 2],
// }
//
// impl Default for Entry {
//     fn default() -> Self {
//         Self {
//             content_id: 0,
//             secondary_content_id: 0,
//             data_id: 0,
//             orientation: [0, 0],
//         }
//     }
// }

#[derive(Default, Copy, Clone)]
pub struct Occluder(u8);

impl Occluder {
    pub fn new(x_neg: bool, x_pos: bool, y_neg: bool, y_pos: bool, z_neg: bool, z_pos: bool) -> Occluder {
        Occluder(
            ((x_neg as u8) << (Facing::NegativeX as u8))
                | ((x_pos as u8) << (Facing::PositiveX as u8))
                | ((y_neg as u8) << (Facing::NegativeY as u8))
                | ((y_pos as u8) << (Facing::PositiveY as u8))
                | ((z_neg as u8) << (Facing::NegativeZ as u8))
                | ((z_pos as u8) << (Facing::PositiveZ as u8)),
        )
    }

    pub fn occludes_side(&self, facing: Facing) -> bool {
        ((self.0 >> (facing as u8)) & 1) == 1
    }
}

impl BitOr for Occluder {
    type Output = Self;

    fn bitor(mut self, rhs: Self) -> Self::Output {
        self.0 |= rhs.0;
        self
    }
}

impl BitOrAssign for Occluder {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = *self | rhs;
    }
}

struct Sector {
    block_storage: EntityStorage,
    block_map: Box<[[[EntityId; SECTOR_SIZE]; SECTOR_SIZE]; SECTOR_SIZE]>,
    block_props: Box<[[[BlockProps; SECTOR_SIZE]; SECTOR_SIZE]; SECTOR_SIZE]>,
    occluders: Box<[[[Occluder; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]>,
    changed: bool,
    side_changed: [bool; 6],
    vertex_mesh: renderer::VertexMesh<Vertex, ()>,
}

impl Sector {
    fn new(layout: &EntityStorageLayout) -> Self {
        Self {
            block_storage: EntityStorage::new(layout),
            block_map: Default::default(),
            block_props: Default::default(),
            occluders: Default::default(),
            changed: false,
            side_changed: [false; 6],
            vertex_mesh: Default::default(),
        }
    }

    const fn entry_index(x: u32, y: u32, z: u32) -> usize {
        SECTOR_SIZE * SECTOR_SIZE * x as usize + SECTOR_SIZE * y as usize + z as usize
    }

    fn set_block(&mut self, pos: U32Vec3, archetype_id: u32) -> BlockDataBuilder {
        let pos: Vector3<usize> = glm::convert(pos);
        let entity_id = &mut self.block_map[pos.x][pos.y][pos.z];
        if *entity_id != EntityId::NULL {
            self.block_storage.remove(entity_id);
        }
        let entity_builder = self.block_storage.add_entity(archetype_id);

        self.occluders[pos.x + 1][pos.y + 1][pos.z + 1] = Occluder::new(true, true, true, true, true, true); // TODO

        let mut side_changed = self.side_changed;
        side_changed[0] |= pos.x == 0;
        side_changed[1] |= pos.x == 15;
        side_changed[2] |= pos.y == 0;
        side_changed[3] |= pos.y == 15;
        side_changed[4] |= pos.z == 0;
        side_changed[5] |= pos.z == 15;

        self.side_changed = side_changed;
        self.changed = true;

        let block_props = &mut self.block_props[pos.x][pos.y][pos.z];

        BlockDataBuilder {
            entity_builder,
            entity_id,
            block_props,
        }
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
    block_props: &'a mut BlockProps,
}

impl BlockDataBuilder<'_> {
    pub fn props(mut self, props: BlockProps) -> Self {
        *self.block_props = props;
        self
    }

    pub fn with<C: 'static>(mut self, comp: C) -> Self {
        self.entity_builder = self.entity_builder.with::<C>(comp);
        self
    }

    pub fn build(self) {
        *self.entity_id = self.entity_builder.build();
    }
}

pub struct Cluster {
    block_registry: Arc<GameRegistry>,
    sectors: [Sector; VOLUME_IN_SECTORS],
    entry_size: u32,
    sides_changed: bool,
    device: Arc<vkw::Device>,
}

impl Cluster {
    pub fn entry_size(&self) -> u32 {
        self.entry_size
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

    pub fn set_block(&mut self, pos: U32Vec3, archetype_id: u32) -> BlockDataBuilder {
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

        self.sides_changed |=
            glm::any(&pos.map(|v| v == 0)) || glm::any(&pos.map(|v| v == (SIZE as u32 - 1)));

        sector.set_block(pos, archetype_id)
    }

    pub fn paste_outer_side_occlusion(&mut self, side_cluster: &mut Cluster, side_offset: I32Vec3) {
        fn map_sector_pos(d: I32Vec3, k: u32, l: u32) -> U32Vec3 {
            let dx = (d.x != 0) as u32;
            let dy = (d.y != 0) as u32;
            let dz = (d.z != 0) as u32;
            let x = k * dy + k * dz + (SIZE_IN_SECTORS as u32 - 1) * (d.x > 0) as u32;
            let y = k * dx + l * dz + (SIZE_IN_SECTORS as u32 - 1) * (d.y > 0) as u32;
            let z = l * dx + l * dy + (SIZE_IN_SECTORS as u32 - 1) * (d.z > 0) as u32;
            U32Vec3::new(x, y, z)
        }
        fn map_pos(d: I32Vec3, k: usize, l: usize, v: usize) -> Vector3<usize> {
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;
            let x = k * dy + k * dz + v * (d.x > 0) as usize;
            let y = k * dx + l * dz + v * (d.y > 0) as usize;
            let z = l * dx + l * dy + v * (d.z > 0) as usize;
            Vector3::new(x, y, z)
        }

        let lb = -(SIZE as i32 * side_cluster.entry_size as i32);
        let rb = SIZE as i32 * self.entry_size as i32;
        let dir = side_offset.map(|v| (v == rb) as i32 - (v == lb) as i32);

        if side_cluster.entry_size == self.entry_size {
            for i in 0..SIZE_IN_SECTORS {
                for j in 0..SIZE_IN_SECTORS {
                    let dst_sector = &mut self.sectors[sector_index(map_sector_pos(dir, i as u32, j as u32))];
                    let src_sector =
                        &mut side_cluster.sectors[sector_index(map_sector_pos(-dir, i as u32, j as u32))];

                    for k in 0..SECTOR_SIZE {
                        for l in 0..SECTOR_SIZE {
                            let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SECTOR_SIZE - 1);
                            let src_p = map_pos(-dir, k, l, SECTOR_SIZE - 2).add_scalar(1);

                            dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                                src_sector.occluders[src_p.x][src_p.y][src_p.z];
                        }
                    }
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
                        &mut side_cluster.sectors[sector_index(map_sector_pos(-dir, i as u32, j as u32))];

                    let ks = (i % 2) * SECTOR_SIZE / 2;
                    let ls = (j % 2) * SECTOR_SIZE / 2;

                    for k in (0..SECTOR_SIZE).step_by(2) {
                        for l in (0..SECTOR_SIZE).step_by(2) {
                            let dst_p = map_pos(dir, ks + k / 2 + 1, ls + l / 2 + 1, ALIGNED_SECTOR_SIZE - 1);
                            let mut occluder = Occluder::default();

                            for k2 in 0..2 {
                                for l2 in 0..2 {
                                    let src_p = map_pos(-dir, k + k2, l + l2, SECTOR_SIZE - 2).add_scalar(1);
                                    occluder |= src_sector.occluders[src_p.x][src_p.y][src_p.z];
                                }
                            }

                            dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] = occluder;
                        }
                    }
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
                    let src_sector = &mut side_cluster.sectors
                        [sector_index(map_sector_pos(-dir, is + i as u32 / 2, js + j as u32 / 2))];

                    let ks = (i % 2) * SECTOR_SIZE / 2;
                    let ls = (j % 2) * SECTOR_SIZE / 2;

                    for k in 0..SECTOR_SIZE {
                        for l in 0..SECTOR_SIZE {
                            let dst_p = map_pos(dir, k + 1, l + 1, ALIGNED_SECTOR_SIZE - 1);
                            let src_p = map_pos(-dir, ks + k / 2, ls + l / 2, SECTOR_SIZE - 2).add_scalar(1);

                            dst_sector.occluders[dst_p.x][dst_p.y][dst_p.z] =
                                src_sector.occluders[src_p.x][src_p.y][src_p.z];
                        }
                    }
                }
            }
        }
    }

    fn update_inner_side_occluders(&mut self) {
        fn map_dst_pos(facing: Facing, k: usize, l: usize) -> (usize, usize, usize) {
            let d = facing.direction();
            let dx = (d.x != 0) as usize;
            let dy = (d.y != 0) as usize;
            let dz = (d.z != 0) as usize;

            let (k, l) = (k + 1, l + 1);
            let x = k * dy + k * dz + (ALIGNED_SECTOR_SIZE - 1) * (d.x > 0) as usize;
            let y = k * dx + l * dz + (ALIGNED_SECTOR_SIZE - 1) * (d.y > 0) as usize;
            let z = l * dx + l * dy + (ALIGNED_SECTOR_SIZE - 1) * (d.z > 0) as usize;

            (x, y, z)
        }

        macro_rules! side_loop {
            ($sector: ident, $sectors: ident, $pos: ident, $facing: expr) => {
                let j = $facing as usize;
                let rel = ($pos + $facing.direction());

                if $sector.side_changed[j]
                    && rel >= I32Vec3::from_element(0)
                    && rel < I32Vec3::from_element(SIZE_IN_SECTORS as i32)
                {
                    let side_sector = &mut $sectors[sector_index(glm::try_convert(rel).unwrap())];
                    let facing_m = $facing.mirror();

                    for k in 0..SECTOR_SIZE {
                        for l in 0..SECTOR_SIZE {
                            let p = map_dst_pos(facing_m, k, l);
                            side_sector.occluders[p.0][p.1][p.2] =
                                Occluder::new(true, true, true, true, true, true);
                        }
                    }

                    $sector.side_changed[j] = false;
                }
            };
        }

        for i in 0..VOLUME_IN_SECTORS {
            let (sector, mut sectors) = self.sectors.split_mid_mut(i).unwrap();
            let pos: I32Vec3 = glm::convert(sector_pos(i));

            side_loop!(sector, sectors, pos, Facing::NegativeX);
            side_loop!(sector, sectors, pos, Facing::PositiveX);
            side_loop!(sector, sectors, pos, Facing::NegativeY);
            side_loop!(sector, sectors, pos, Facing::PositiveY);
            side_loop!(sector, sectors, pos, Facing::NegativeZ);
            side_loop!(sector, sectors, pos, Facing::PositiveZ);
        }
    }

    fn triangulate(&mut self, sector_pos: U32Vec3) -> (Vec<Vertex>, Vec<u32>) {
        let sector = &mut self.sectors[sector_index(sector_pos)];
        let scale = self.entry_size as f32;
        let mut vertices = Vec::<Vertex>::with_capacity(SECTOR_VOLUME * 8);

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
                    let props = &sector.block_props[x][y][z];
                    let model = self
                        .block_registry
                        .get_textured_block_model(props.textured_model_id())
                        .unwrap();

                    add_vertices(&mut vertices, posf, scale, model.get_inner_quads());

                    for i in 0..6 {
                        let facing = Facing::from_u8(i as u8);
                        let rel = (pos + facing.direction()).add_scalar(1);

                        let occludes = sector.occluders[rel.x as usize][rel.y as usize][rel.z as usize]
                            .occludes_side(facing.mirror());

                        if !occludes {
                            add_vertices(&mut vertices, posf, scale, model.get_quads_by_facing(facing));
                            // TODO: Ambient occlusion
                        }
                    }
                }
            }
        }

        let mut indices = vec![u32::MAX; vertices.len() / 4 * 6];

        for i in (0..indices.len()).step_by(6) {
            let ind = (i / 6 * 4) as u32;
            indices[i] = ind;
            indices[i + 1] = ind + 2;
            indices[i + 2] = ind + 1;
            indices[i + 3] = ind + 2;
            indices[i + 4] = ind + 3;
            indices[i + 5] = ind + 1;
        }

        (vertices, indices)
    }

    pub fn update_mesh(&mut self, simplification_factor: f32) {
        self.update_inner_side_occluders();

        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    let sector_index = sector_index(U32Vec3::new(x as u32, y as u32, z as u32));
                    let sector_changed = self.sectors[sector_index].changed;

                    if sector_changed {
                        let sector = &self.sectors[sector_index];
                        let mut sector_vertices = Vec::with_capacity(SECTOR_VOLUME);
                        let mut sector_indices = Vec::with_capacity(SECTOR_VOLUME);

                        let (mut temp_vertices, temp_indices) =
                            self.triangulate(U32Vec3::new(x as u32, y as u32, z as u32));

                        let (temp_vertices, temp_indices) = {
                            let options = mesh_simplifier::Options::new(
                                0.125,
                                10,
                                (512 as f32 * (1.0 - simplification_factor)) as usize,
                                4.0 * self.entry_size as f32,
                                1.0,
                                0.8,
                            );

                            // utils::calc_smooth_mesh_normals(&mut temp_vertices, &temp_indices);
                            // let (mut vertices, indices) =
                            //     mesh_simplifier::simplify(&temp_vertices, &temp_indices, &options);
                            // utils::calc_smooth_mesh_normals(&mut vertices, &indices);
                            // (vertices, indices)
                            (temp_vertices, temp_indices)
                        };

                        sector_vertices.extend(temp_vertices);
                        sector_indices.extend(temp_indices);

                        let sector = &mut self.sectors[sector_index];

                        sector.vertex_mesh = self
                            .device
                            .create_vertex_mesh(&sector_vertices, Some(&sector_indices))
                            .unwrap();
                        sector.changed = false;
                    }
                }
            }
        }
    }
}

pub struct UpdateSystemData<'a> {
    pub mat_pipeline: Arc<MaterialPipeline>,
    pub entities: &'a mut scene::Entities,
    pub transform: scene::ComponentStorageMut<'a, component::Transform>,
    pub renderer: scene::ComponentStorageMut<'a, component::Renderer>,
    pub vertex_mesh: scene::ComponentStorageMut<'a, component::VertexMesh>,
    pub children: scene::ComponentStorageMut<'a, component::Children>,
}

impl Cluster {
    pub fn update_renderable(&self, entity: u32, data: &mut UpdateSystemData) {
        let transform_comps = &mut data.transform;
        let renderer_comps = &mut data.renderer;
        let vertex_mesh_comps = &mut data.vertex_mesh;
        let children_comps = &mut data.children;
        let entities = &mut data.entities;

        let is_children_empty = if let Some(children) = children_comps.get(entity) {
            children.0.is_empty()
        } else {
            children_comps.set(entity, component::Children::default());
            true
        };

        if is_children_empty {
            let children = children_comps.get_mut(entity).unwrap();
            let sector_count = SIZE_IN_SECTORS * SIZE_IN_SECTORS * SIZE_IN_SECTORS;

            children.0 = (0..sector_count).into_iter().map(|_| entities.create()).collect();

            for (i, &ent) in children.0.iter().enumerate() {
                let p = index_1d_to_3d(i, SIZE_IN_SECTORS);
                let node_size = self.entry_size as usize;

                transform_comps.set(
                    ent,
                    component::Transform::new(
                        glm::convert(Vector3::new(p[0], p[1], p[2]) * SECTOR_SIZE * node_size),
                        Vec3::default(),
                        Vec3::from_element(1.0),
                    ),
                );

                let renderer = component::Renderer::new(&self.device, &data.mat_pipeline, false);
                renderer_comps.set(ent, renderer);
            }
        }

        let children = children_comps.get(entity).unwrap();

        for (i, &ent) in children.0.iter().enumerate() {
            let sector = &self.sectors[i];

            vertex_mesh_comps.set(ent, component::VertexMesh::new(&sector.vertex_mesh.raw()));
        }
    }
}

pub fn new(block_registry: &Arc<GameRegistry>, device: &Arc<vkw::Device>, node_size: u32) -> Cluster {
    let layout = block_registry.cluster_layout();
    let sectors: Vec<Sector> = (0..VOLUME_IN_SECTORS).map(|_| Sector::new(&layout)).collect();

    Cluster {
        block_registry: Arc::clone(block_registry),
        sectors: sectors.try_into().ok().unwrap(),
        entry_size: node_size,
        sides_changed: false,
        device: Arc::clone(device),
    }
}