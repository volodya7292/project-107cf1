use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::vertex_mesh::VertexMeshCreate;
use crate::renderer::{component, scene};
use crate::utils::{mesh_simplifier, HashMap};
use crate::{renderer, utils};
use dual_contouring as dc;
use nalgebra as na;
use smallvec::smallvec;
use std::collections::hash_map;
use std::convert::TryInto;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;

const SECTOR_SIZE: usize = 16;
const ALIGNED_SECTOR_SIZE: usize = SECTOR_SIZE + 1;
const SIZE_IN_SECTORS: usize = 4;
pub const SIZE: usize = SECTOR_SIZE * SIZE_IN_SECTORS;
pub const VOLUME: usize = SIZE * SIZE * SIZE;
const SECTOR_VOLUME: usize = SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE;
const ALIGNED_SECTOR_VOLUME: usize = ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE;

fn index_3d_to_1d(p: [u8; 3], ds: u32) -> u32 {
    (p[2] as u32) + (p[1] as u32) * ds + (p[0] as u32) * ds * ds
}

fn index_3d_to_1d_inv(p: [u8; 3], ds: u32) -> u32 {
    (p[0] as u32) + (p[1] as u32) * ds + (p[2] as u32) * ds * ds
}

fn index_1d_to_3d(i: usize, ds: usize) -> [usize; 3] {
    let ds_sqr = ds * ds;
    let i_2d = i % ds_sqr;
    [i / ds_sqr, i_2d / ds, i_2d % ds]
}

struct ContentType(u8);

impl ContentType {
    const SOLID: ContentType = ContentType(0);
    const BINARY_TRANSPARENT: ContentType = ContentType(1);
    const TRANSLUCENT: ContentType = ContentType(2);
}

pub struct Entry {
    content_id: u16,
    secondary_content_id: u16,
    data_id: u16,
    orientation: [u8; 2],
}

impl Default for Entry {
    fn default() -> Self {
        Self {
            content_id: 0,
            secondary_content_id: 0,
            data_id: 0,
            orientation: [0, 0],
        }
    }
}

#[derive(Copy, Clone, Default)]
#[repr(C)]
pub struct Vertex {
    position: na::Vector3<f32>,
    normal: na::Vector3<f32>,
    material_ids: na::Vector4<u32>,
}
vertex_impl!(Vertex, position, normal, material_ids);

struct Sector {
    entries: Box<[[[Entry; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]>,
    changed: bool,
    seam_influence_changed: bool,
    seam_changed: bool,
    vertex_mesh: renderer::VertexMesh<Vertex>,
}

impl Sector {}

impl Default for Sector {
    fn default() -> Self {
        Self {
            entries: Default::default(),
            changed: false,
            seam_influence_changed: false,
            seam_changed: false,
            vertex_mesh: Default::default(),
        }
    }
}

pub struct Cluster {
    sectors: [[[Sector; SIZE_IN_SECTORS]; SIZE_IN_SECTORS]; SIZE_IN_SECTORS],
    entry_size: u32,
    device: Arc<vkw::Device>,
}

impl Cluster {
    pub fn calc_sector_position(cell_pos: [u8; 3]) -> [usize; 3] {
        [
            (cell_pos[0] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
            (cell_pos[1] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
            (cell_pos[2] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
        ]
    }

    pub fn entry_size(&self) -> u32 {
        self.entry_size
    }

    fn triangulate(&mut self, sector_pos: [u32; 3]) -> (Vec<Vertex>, Vec<u32>) {
        let sector =
            &mut self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize];

        unimplemented!()
    }

    pub fn update_mesh(&mut self, simplification_factor: f32) {
        // Collect vertices & indices
        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    let sector_changed = self.sectors[x][y][z].changed;

                    if sector_changed {
                        let sector = &self.sectors[x][y][z];
                        let mut sector_vertices = Vec::with_capacity(SECTOR_VOLUME);
                        let mut sector_indices = Vec::with_capacity(SECTOR_VOLUME);

                        let (mut temp_vertices, temp_indices) =
                            self.triangulate([x as u32, y as u32, z as u32]);

                        let (temp_vertices, temp_indices) = {
                            let options = mesh_simplifier::Options::new(
                                0.125,
                                10,
                                (512 as f32 * (1.0 - simplification_factor)) as usize,
                                4.0 * self.entry_size as f32,
                                1.0,
                                0.8,
                            );

                            utils::calc_smooth_mesh_normals(&mut temp_vertices, &temp_indices);
                            let (mut vertices, indices) =
                                mesh_simplifier::simplify(&temp_vertices, &temp_indices, &options);
                            utils::calc_smooth_mesh_normals(&mut vertices, &indices);
                            (vertices, indices)
                        };

                        sector_vertices.extend(temp_vertices);
                        sector_indices.extend(temp_indices);

                        let sector = &mut self.sectors[x][y][z];

                        self.sectors[x][y][z].vertex_mesh = self
                            .device
                            .create_vertex_mesh(&sector_vertices, Some(&sector_indices))
                            .unwrap();
                        self.sectors[x][y][z].changed = false;
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
                        na::convert(na::Vector3::new(p[0], p[1], p[2]) * SECTOR_SIZE * node_size),
                        na::Vector3::default(),
                        na::Vector3::from_element(1.0),
                    ),
                );

                let renderer = component::Renderer::new(&self.device, &data.mat_pipeline, false);
                renderer_comps.set(ent, renderer);
            }
        }

        let children = children_comps.get(entity).unwrap();

        for (i, &ent) in children.0.iter().enumerate() {
            let p = index_1d_to_3d(i, SIZE_IN_SECTORS);
            let sector = &self.sectors[p[0]][p[1]][p[2]];

            vertex_mesh_comps.set(ent, component::VertexMesh::new(&sector.vertex_mesh.raw()));
        }
    }
}

pub fn new(device: &Arc<vkw::Device>, node_size: u32) -> Cluster {
    Cluster {
        sectors: Default::default(),
        entry_size: node_size,
        device: Arc::clone(device),
    }
}
