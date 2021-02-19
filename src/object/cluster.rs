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
const ALIGNED_SECTOR_SIZE: usize = SECTOR_SIZE + 2;
const SIZE_IN_SECTORS: usize = 4;
pub const SIZE: usize = SECTOR_SIZE * SIZE_IN_SECTORS;
pub const VOLUME: usize = SIZE * SIZE * SIZE;
const ALIGNED_SIZE: usize = SIZE + 2;
pub const MAX_CELL_LAYERS: usize = 4; // MAX: 255
const SECTOR_VOLUME: usize = SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE;
const ALIGNED_SECTOR_VOLUME: usize = ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE;
const ALIGNED_SECTOR_MAX_CELLS: usize = ALIGNED_SECTOR_VOLUME * MAX_CELL_LAYERS;
pub const ISO_VALUE_NORM: f32 = 0.5;
const ISO_VALUE_INT: i16 = (ISO_VALUE_NORM * 255.0) as i16;

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

pub fn calc_density_index(head_index: u32, layer_count: u8) -> u32 {
    head_index | ((layer_count as u32) << 24)
}

#[derive(Debug, Default, Copy, Clone)]
pub struct DensityPoint {
    pub(crate) density: u8,
    pub(crate) material: u16,
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
    indices: Box<[[[u32; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]>,
    densities: Vec<DensityPoint>,
    layer_count: u32,
    indices_changed: bool,
    changed: bool,
    seam_influence_changed: bool,
    seam_changed: bool,
    vertex_count: u32,
    index_count: u32,
    vertex_mesh: renderer::VertexMesh<Vertex>,
}

#[derive(Copy, Clone)]
pub struct DensityPointInfo {
    /// [x, y, z, layer index]
    pub(crate) pos: [u8; 4],
    pub(crate) point: DensityPoint,
}

type EncodedNode = dc::octree::EncodedNode<[PointData; 8]>;

impl Sector {
    /// pos: [x, y, z, layer index]
    fn get_density(&self, pos: [u8; 4]) -> Option<DensityPoint> {
        let index = *self
            .indices
            .get(pos[0] as usize)?
            .get(pos[1] as usize)?
            .get(pos[2] as usize)?;
        let head_index = index & 0x00ffffff;
        let layer_count = (index & 0xff000000) >> 24;

        if pos[3] >= layer_count as u8 {
            return None;
        }
        Some(self.densities[head_index as usize + pos[3] as usize])
    }

    /// pos: [x, y, z, layer index]
    fn get_density_layers(&self, pos: [u8; 3], out: &mut [DensityPoint]) -> u8 {
        let index = self.indices[pos[0] as usize][pos[1] as usize][pos[2] as usize];
        let head_index = index & 0x00ffffff;
        let layer_count = (index & 0xff000000) >> 24;
        let read_count = out.len().min(layer_count as usize);

        for i in 0..read_count {
            out[i] = self.densities[head_index as usize + i];
        }

        return read_count as u8;
    }

    /// All points must be unique.
    fn set_densities(&mut self, points: &mut [DensityPointInfo]) {
        // Sort points by position starting from [0,,] to [sector_SIZE,,]
        points.sort_unstable_by(|a, b| {
            let a_dist = index_3d_to_1d((&a.pos[..3]).try_into().unwrap(), ALIGNED_SECTOR_SIZE as u32);
            let b_dist = index_3d_to_1d((&b.pos[..3]).try_into().unwrap(), ALIGNED_SECTOR_SIZE as u32);
            a_dist.cmp(&b_dist)
        });

        let mut offset = 0u32;
        let mut temp_offset = 0u32;
        let mut last_pos_1d = 0u32;
        let mut changed = false;
        let mut indices_changed = true;

        for point_info in points {
            let pos = [
                point_info.pos[0] as usize,
                point_info.pos[1] as usize,
                point_info.pos[2] as usize,
            ];

            let index = &mut self.indices[pos[0]][pos[1]][pos[2]];
            let head_index = *index & 0x00ffffff;
            let layer_count = (*index & 0xff000000) >> 24;

            let pos_1d = index_3d_to_1d(
                (&point_info.pos[..3]).try_into().unwrap(),
                ALIGNED_SECTOR_SIZE as u32,
            );
            if pos_1d != last_pos_1d {
                offset += temp_offset;
                temp_offset = 0;
                last_pos_1d = pos_1d;
            }

            let insert_index = (head_index + point_info.pos[3] as u32 + offset) as usize;

            if point_info.pos[3] < MAX_CELL_LAYERS as u8 {
                if point_info.pos[3] < layer_count as u8 {
                    self.densities[insert_index] = point_info.point;
                } else if point_info.pos[3] == layer_count as u8 {
                    // Add a new layer
                    self.densities.insert(insert_index, point_info.point);
                    *index = head_index | ((layer_count + 1) << 24);
                    temp_offset += 1;
                    indices_changed = true;
                }
                changed = true;
            }
        }

        self.indices_changed |= indices_changed;
        self.changed |= changed;

        self.update_indices();
    }

    fn update_indices(&mut self) {
        if !self.indices_changed {
            return;
        }
        self.indices_changed = false;

        let mut max_layer_count = 0;
        let mut offset = 0u32;

        for x in 0..ALIGNED_SECTOR_SIZE {
            for y in 0..ALIGNED_SECTOR_SIZE {
                for z in 0..ALIGNED_SECTOR_SIZE {
                    let index = &mut self.indices[x][y][z];
                    let layer_count = (*index & 0xff000000) >> 24;

                    *index = offset | (layer_count << 24);
                    offset += layer_count;

                    max_layer_count = max_layer_count.max(layer_count);
                }
            }
        }

        self.layer_count = max_layer_count;
    }
}

impl Default for Sector {
    fn default() -> Self {
        Self {
            indices: Box::new([[[0; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]),
            densities: vec![],
            layer_count: 0,
            indices_changed: false,
            changed: false,
            seam_influence_changed: false,
            seam_changed: false,
            vertex_count: 0,
            index_count: 0,
            vertex_mesh: Default::default(),
        }
    }
}

#[derive(Copy, Clone, Default)]
pub struct PointData {
    material: u16,
}

pub struct Cluster {
    sectors: [[[Sector; SIZE_IN_SECTORS]; SIZE_IN_SECTORS]; SIZE_IN_SECTORS],
    node_size: u32,
    seam_nodes_cache: HashMap<u32, Vec<dc::octree::LeafNode<dc::contour::NodeDataDiscrete<PointData>>>>,
    // layer -> node cache
    nodes_buffer: Option<vkw::DeviceBuffer>,
    device: Arc<vkw::Device>,
}

#[derive(Default)]
struct SeamLayer {
    /// original-sized nodes: size = { seam.node_size or (seam.node_size / 2) or (seam.node_size * 2) }
    original_nodes: Vec<dc::octree::LeafNode<dc::contour::NodeDataDiscrete<PointData>>>,
    /// nodes are normalized: size = { seam.node_size or (seam.node_size / 2) }
    normalized_nodes: Vec<dc::octree::LeafNode<dc::contour::NodeDataDiscrete<PointData>>>,
}

pub struct Seam {
    /// layer -> node cache
    nodes: HashMap<u32, SeamLayer>,
    node_size: u32,
}

impl Seam {
    pub fn new(node_size: u32) -> Seam {
        Seam {
            nodes: HashMap::with_capacity(MAX_CELL_LAYERS),
            node_size,
        }
    }

    /// Trilinear interpolation
    fn interpolate_unit(d: &[f32; 8], p: na::Vector3<f32>) -> f32 {
        let xm = na::Matrix1x2::new(1.0 - p.x, p.x);
        let ym = na::Matrix2x1::new(1.0 - p.y, p.y);
        let zm = na::Matrix1x2::new(1.0 - p.z, p.z);

        let fm0 = na::Matrix2::new(d[0], d[2], d[4], d[6]);
        let fm1 = na::Matrix2::new(d[1], d[3], d[5], d[7]);

        (zm * na::Matrix2x1::new((xm * fm0 * ym)[(0, 0)], (xm * fm1 * ym)[(0, 0)]))[(0, 0)]
    }

    pub fn insert(&mut self, cluster: &mut Cluster, offset: na::Vector3<i32>) {
        let cluster_seam_nodes = cluster.collect_nodes_for_seams();

        let bound = ((SIZE as u32) * self.node_size) as i32;

        let mut densities = [[[0.0_f32; 3]; 3]; 3];
        let mut mat_data = [[[PointData::default(); 3]; 3]; 3];

        for (i, nodes) in cluster_seam_nodes {
            let self_nodes = self.nodes.entry(*i).or_insert(SeamLayer {
                original_nodes: Vec::with_capacity(SIZE * SIZE * 8),
                normalized_nodes: Vec::with_capacity(SIZE * SIZE * 8),
            });

            for node in nodes {
                let pos = na::convert::<na::Vector3<u32>, na::Vector3<i32>>(*node.position()) + offset;
                let size = node.size();

                if (size != self.node_size && (size != self.node_size / 2) && (size != self.node_size * 2))
                    || (pos.x < bound && pos.y < bound && pos.z < bound)
                    || (pos.x > bound || pos.y > bound || pos.z > bound)
                    || (pos.x < 0 || pos.y < 0 || pos.z < 0)
                {
                    continue;
                }

                self_nodes.original_nodes.push(dc::octree::LeafNode::new(
                    na::try_convert(pos).unwrap(),
                    node.size(),
                    *node.data(),
                ));

                if size == self.node_size * 2 {
                    let data = node.data();

                    for x in 0..3 {
                        for y in 0..3 {
                            for z in 0..3 {
                                densities[x][y][z] = Self::interpolate_unit(
                                    &data.densities,
                                    na::Vector3::new(x as f32, y as f32, z as f32) * 0.5,
                                );
                                mat_data[x][y][z] = data.data[x / 2 * 4 + y / 2 * 2 + z / 2];
                            }
                        }
                    }

                    for x in 0..2 {
                        for y in 0..2 {
                            for z in 0..2 {
                                let d = [
                                    densities[x + 0][y + 0][z + 0],
                                    densities[x + 0][y + 0][z + 1],
                                    densities[x + 0][y + 1][z + 0],
                                    densities[x + 0][y + 1][z + 1],
                                    densities[x + 1][y + 0][z + 0],
                                    densities[x + 1][y + 0][z + 1],
                                    densities[x + 1][y + 1][z + 0],
                                    densities[x + 1][y + 1][z + 1],
                                ];
                                let mats = [
                                    mat_data[x + 0][y + 0][z + 0],
                                    mat_data[x + 0][y + 0][z + 1],
                                    mat_data[x + 0][y + 1][z + 0],
                                    mat_data[x + 0][y + 1][z + 1],
                                    mat_data[x + 1][y + 0][z + 0],
                                    mat_data[x + 1][y + 0][z + 1],
                                    mat_data[x + 1][y + 1][z + 0],
                                    mat_data[x + 1][y + 1][z + 1],
                                ];

                                let new_pos = pos
                                    + na::Vector3::new(x as i32, y as i32, z as i32)
                                        * (self.node_size as i32);

                                if new_pos.x > bound || new_pos.y > bound || new_pos.z > bound {
                                    continue;
                                }

                                let data = dc::contour::NodeDataDiscrete::new(d, ISO_VALUE_NORM, mats);

                                self_nodes.normalized_nodes.push(dc::octree::LeafNode::new(
                                    na::try_convert(new_pos).unwrap(),
                                    self.node_size,
                                    data,
                                ));
                            }
                        }
                    }
                } else {
                    self_nodes.normalized_nodes.push(dc::octree::LeafNode::new(
                        na::try_convert(pos).unwrap(),
                        node.size(),
                        *node.data(),
                    ));
                }
            }
        }
    }
}

impl Cluster {
    pub fn calc_sector_position(cell_pos: [u8; 3]) -> [usize; 3] {
        [
            (cell_pos[0] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
            (cell_pos[1] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
            (cell_pos[2] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
        ]
    }

    pub fn node_size(&self) -> u32 {
        self.node_size
    }

    pub fn layer_count(&self) -> u32 {
        let mut layer_count = 0;

        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    layer_count = layer_count.max(self.sectors[x][y][z].layer_count);
                }
            }
        }

        layer_count
    }

    /// pos: [x, y, z, layer index]
    pub fn get_density(&self, pos: [u8; 4]) -> Option<DensityPoint> {
        let sector_pos = Self::calc_sector_position((&pos[..3]).try_into().unwrap());
        let sector = &self.sectors[sector_pos[0]][sector_pos[1]][sector_pos[2]];

        sector.get_density([
            pos[0] % SECTOR_SIZE as u8,
            pos[1] % SECTOR_SIZE as u8,
            pos[2] % SECTOR_SIZE as u8,
            pos[3],
        ])
    }

    /// pos: [x, y, z, layer index]
    pub fn get_density_layers(&self, pos: [u8; 3], out: &mut [DensityPoint]) -> u8 {
        let sector_pos = Self::calc_sector_position((&pos[..3]).try_into().unwrap());
        let sector = &self.sectors[sector_pos[0]][sector_pos[1]][sector_pos[2]];
        sector.get_density_layers(
            [
                pos[0] - ((SECTOR_SIZE * sector_pos[0]) as u8),
                pos[1] - ((SECTOR_SIZE * sector_pos[1]) as u8),
                pos[2] - ((SECTOR_SIZE * sector_pos[2]) as u8),
            ],
            out,
        )
    }

    /// All points must be unique.
    pub fn set_densities(&mut self, points: &[DensityPointInfo]) {
        // Sort points by position starting from [0,,] to [SIZE,,]
        let mut points = points.to_vec();
        points.sort_unstable_by(|a, b| {
            let a_dist = index_3d_to_1d((&a.pos[..3]).try_into().unwrap(), ALIGNED_SIZE as u32);
            let b_dist = index_3d_to_1d((&b.pos[..3]).try_into().unwrap(), ALIGNED_SIZE as u32);

            a_dist.cmp(&b_dist)
        });

        // Offset for each sector for density insert (density_indices isn't updated every insertion to densities)
        // (offset, temp_offset, last_pos_1d)
        let mut offsets = [[[(0u32, 0u32, 0u32); SIZE_IN_SECTORS]; SIZE_IN_SECTORS]; SIZE_IN_SECTORS];

        for point_info in &points {
            let sector_pos = Self::calc_sector_position((&point_info.pos[..3]).try_into().unwrap());
            let sector = &mut self.sectors[sector_pos[0]][sector_pos[1]][sector_pos[2]];

            let pos_in_sector = [
                (point_info.pos[0] as usize - SECTOR_SIZE * sector_pos[0]),
                (point_info.pos[1] as usize - SECTOR_SIZE * sector_pos[1]),
                (point_info.pos[2] as usize - SECTOR_SIZE * sector_pos[2]),
            ];

            let index = &mut sector.indices[pos_in_sector[0]][pos_in_sector[1]][pos_in_sector[2]];
            let head_index = *index & 0x00ffffff;
            let layer_count = (*index & 0xff000000) >> 24;

            let offset = &mut offsets[sector_pos[0]][sector_pos[1]][sector_pos[2]];

            let pos_1d = index_3d_to_1d((&point_info.pos[..3]).try_into().unwrap(), ALIGNED_SIZE as u32);
            if pos_1d != offset.2 {
                offset.0 += offset.1;
                offset.1 = 0;
                offset.2 = pos_1d;
            }

            let insert_index = (head_index + point_info.pos[3] as u32 + offset.0) as usize;

            if point_info.pos[3] >= MAX_CELL_LAYERS as u8 {
                continue;
            }

            if point_info.pos[3] < layer_count as u8 {
                sector.densities[insert_index] = point_info.point;
            } else if point_info.pos[3] == layer_count as u8 {
                // Add a new layer
                sector.densities.insert(insert_index, point_info.point);
                *index = head_index | ((layer_count + 1) << 24);
                offset.1 += 1;
                sector.indices_changed = true;
            }
            sector.changed = true;
        }

        // Update indices to account for inserted densities
        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    let mut sector = &mut self.sectors[x][y][z];
                    sector.update_indices();
                    sector.seam_influence_changed = true;
                }
            }
        }

        self.seam_nodes_cache.clear();
    }

    fn update_sector_seams(&mut self, sector_pos: [u8; 3]) {
        {
            let sector =
                &mut self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize];
            if !sector.seam_changed {
                return;
            }
            sector.seam_changed = false;
        }

        let mut temp_density_infos = Vec::<DensityPointInfo>::with_capacity(SECTOR_SIZE * SECTOR_SIZE * 8);
        let mut temp_density = [DensityPoint::default(); MAX_CELL_LAYERS];

        // Right side
        if sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8 {
            let sector =
                &self.sectors[sector_pos[0] as usize + 1][sector_pos[1] as usize][sector_pos[2] as usize];

            for x in 0..2 {
                for y in 0..SECTOR_SIZE {
                    for z in 0..SECTOR_SIZE {
                        let count = sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [SECTOR_SIZE as u8 + x, y as u8, z as u8, i],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        // Top side
        if sector_pos[1] < (SIZE_IN_SECTORS - 1) as u8 {
            let sector =
                &self.sectors[sector_pos[0] as usize][sector_pos[1] as usize + 1][sector_pos[2] as usize];

            for x in 0..SECTOR_SIZE {
                for y in 0..2 {
                    for z in 0..SECTOR_SIZE {
                        let count = sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [x as u8, SECTOR_SIZE as u8 + y, z as u8, i],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        // Back side
        if sector_pos[2] < (SIZE_IN_SECTORS - 1) as u8 {
            let sector =
                &self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize + 1];

            for x in 0..SECTOR_SIZE {
                for y in 0..SECTOR_SIZE {
                    for z in 0..2 {
                        let count = sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [x as u8, y as u8, SECTOR_SIZE as u8 + z, i],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        // Right-Top edge
        let c0 = sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8;
        let c1 = sector_pos[1] < (SIZE_IN_SECTORS - 1) as u8;

        if c0 || c1 {
            let sector = &self.sectors[sector_pos[0] as usize + c0 as usize]
                [sector_pos[1] as usize + c1 as usize][sector_pos[2] as usize];

            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..ALIGNED_SECTOR_SIZE {
                        let count = sector.get_density_layers(
                            [
                                (SECTOR_SIZE as u8 * (!c0 as u8)) + x,
                                (SECTOR_SIZE as u8 * (!c1 as u8)) + y,
                                z as u8,
                            ],
                            &mut temp_density,
                        );

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [SECTOR_SIZE as u8 + x, SECTOR_SIZE as u8 + y, z as u8, i],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        // Right-Back edge
        let c0 = sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8;
        let c1 = sector_pos[2] < (SIZE_IN_SECTORS - 1) as u8;

        if c0 || c1 {
            let sector = &self.sectors[sector_pos[0] as usize + c0 as usize][sector_pos[1] as usize]
                [sector_pos[2] as usize + c1 as usize];

            for x in 0..2 {
                for y in 0..ALIGNED_SECTOR_SIZE {
                    for z in 0..2 {
                        let count = sector.get_density_layers(
                            [
                                (SECTOR_SIZE as u8 * (!c0 as u8)) + x,
                                y as u8,
                                (SECTOR_SIZE as u8 * (!c1 as u8)) + z,
                            ],
                            &mut temp_density,
                        );

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [SECTOR_SIZE as u8 + x, y as u8, SECTOR_SIZE as u8 + z, i],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        // Top-Back edge
        let c0 = sector_pos[1] < (SIZE_IN_SECTORS - 1) as u8;
        let c1 = sector_pos[2] < (SIZE_IN_SECTORS - 1) as u8;

        if c0 || c1 {
            let sector = &self.sectors[sector_pos[0] as usize][sector_pos[1] as usize + c0 as usize]
                [sector_pos[2] as usize + c1 as usize];

            for x in 0..ALIGNED_SECTOR_SIZE {
                for y in 0..2 {
                    for z in 0..2 {
                        let count = sector.get_density_layers(
                            [
                                x as u8,
                                (SECTOR_SIZE as u8 * (!c0 as u8)) + y,
                                (SECTOR_SIZE as u8 * (!c1 as u8)) + z,
                            ],
                            &mut temp_density,
                        );

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [x as u8, SECTOR_SIZE as u8 + y, SECTOR_SIZE as u8 + z, i],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        // Corner
        if (sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8)
            && (sector_pos[1] < (SIZE_IN_SECTORS - 1) as u8)
            && (sector_pos[2] < (SIZE_IN_SECTORS - 1) as u8)
        {
            let sector = &self.sectors[sector_pos[0] as usize + 1][sector_pos[1] as usize + 1]
                [sector_pos[2] as usize + 1];

            for x in 0..2 {
                for y in 0..2 {
                    for z in 0..2 {
                        let count = sector.get_density_layers([x, y, z], &mut temp_density);

                        for i in 0..count {
                            temp_density_infos.push(DensityPointInfo {
                                pos: [
                                    SECTOR_SIZE as u8 + x,
                                    SECTOR_SIZE as u8 + y,
                                    SECTOR_SIZE as u8 + z,
                                    i,
                                ],
                                point: temp_density[i as usize],
                            });
                        }
                    }
                }
            }
        }

        let sector =
            &mut self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize];
        sector.set_densities(&mut temp_density_infos);
    }

    fn update_seams(&mut self) {
        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    {
                        let sector = &mut self.sectors[x][y][z];
                        if !sector.seam_influence_changed {
                            continue;
                        }
                        sector.seam_influence_changed = false;
                    }

                    for x2 in x.saturating_sub(1)..(x + 1) {
                        for y2 in y.saturating_sub(1)..(y + 1) {
                            for z2 in z.saturating_sub(1)..(z + 1) {
                                self.sectors[x2][y2][z2].seam_changed = true;
                            }
                        }
                    }
                }
            }
        }

        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    self.update_sector_seams([x as u8, y as u8, z as u8]);
                }
            }
        }
    }

    pub fn fill_seam_densities(&mut self, seam: &Seam) {
        let mut points = Vec::<DensityPointInfo>::with_capacity(SIZE * SIZE * 8);

        let bound = (SIZE as u32) * self.node_size;

        for (level, seam_layer) in &seam.nodes {
            for node in &seam_layer.normalized_nodes {
                let pos = node.position();
                let size = node.size();
                let data = node.data();

                let xc = (pos.x == bound) as u32;
                let yc = (pos.y == bound) as u32;
                let zc = (pos.z == bound) as u32;

                for i in 0..2 {
                    for j in 0..2 {
                        let xyz_u = na::Vector3::new(i * yc + i * zc, i * xc + j * zc, j * xc + j * yc);
                        let xyz = pos + xyz_u * size;

                        if (xyz.x % self.node_size == 0)
                            && (xyz.y % self.node_size == 0)
                            && (xyz.z % self.node_size == 0)
                        {
                            let xyz = xyz / self.node_size;
                            let index = (xyz_u.x * 4 + xyz_u.y * 2 + xyz_u.z) as usize;

                            if xyz_u.x < 2 && xyz_u.y < 2 && xyz_u.z < 2 {
                                points.push(DensityPointInfo {
                                    pos: [xyz.x as u8, xyz.y as u8, xyz.z as u8, *level as u8],
                                    point: DensityPoint {
                                        density: (data.densities[index] * 255.0) as u8,
                                        material: data.data[index].material,
                                    },
                                });
                            }
                        }
                    }
                }
            }
        }

        self.set_densities(&points);
    }

    fn triangulate(
        &mut self,
        sector_pos: [u32; 3],
        layer_index: u32,
        seam_layer: &SeamLayer,
    ) -> (Vec<Vertex>, Vec<u32>) {
        self.update_seams();

        // Create density field
        // ------------------------------------------------------

        let sector =
            &mut self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize];
        let densities = &sector.densities;
        let density_indices = &sector.indices;

        let mut field = vec![None; ALIGNED_SECTOR_VOLUME];

        let field_index = |x: usize, y: usize, z: usize| -> usize {
            (x * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE + y * ALIGNED_SECTOR_SIZE + z) as usize
        };

        let seam_bound0 = na::Vector3::new(sector_pos[0] as u32, sector_pos[1] as u32, sector_pos[2] as u32)
            * (SECTOR_SIZE as u32)
            * self.node_size;
        let seam_bound1 = seam_bound0.add_scalar(SECTOR_SIZE as u32 * self.node_size);

        let is_seam_sector = (sector_pos[0] == SIZE_IN_SECTORS as u32 - 1)
            || (sector_pos[1] == SIZE_IN_SECTORS as u32 - 1)
            || (sector_pos[2] == SIZE_IN_SECTORS as u32 - 1);

        // Fill the field with local data
        for x in 0..ALIGNED_SECTOR_SIZE {
            for y in 0..ALIGNED_SECTOR_SIZE {
                for z in 0..ALIGNED_SECTOR_SIZE {
                    let density_index = density_indices[x][y][z];

                    let index = density_index & 0x00ffffff;
                    let count = density_index >> 24;

                    if count > 0 && layer_index < count {
                        let p = densities[index as usize + layer_index as usize];
                        field[field_index(x, y, z)] =
                            Some(((p.density as f32) / 255.0, PointData { material: p.material }));
                    } else {
                        field[field_index(x, y, z)] = None;
                    }
                }
            }
        }

        // Construct octree & generate mesh
        // ------------------------------------------------------

        // Collect vertices & set node vertex indices
        let mut nodes =
            dc::contour::construct_nodes(&field, (SECTOR_SIZE + 1) as u32, self.node_size, ISO_VALUE_NORM);

        if is_seam_sector {
            // Extend with neighbour nodes
            let self_node_size = self.node_size;
            let neighbour_nodes = seam_layer.original_nodes.iter().filter_map(|node| {
                let pos = node.position();

                if (pos.x == seam_bound1.x || pos.y == seam_bound1.y || pos.z == seam_bound1.z)
                    && (pos.x >= seam_bound0.x
                        && pos.x <= seam_bound1.x
                        && pos.y >= seam_bound0.y
                        && pos.y <= seam_bound1.y
                        && pos.z >= seam_bound0.z
                        && pos.z <= seam_bound1.z)
                {
                    let rel_pos = node.position()
                        - (na::Vector3::new(
                            sector_pos[0] as u32,
                            sector_pos[1] as u32,
                            sector_pos[2] as u32,
                        ) * (SECTOR_SIZE as u32)
                            * self_node_size);
                    let data = node.data();

                    if let Some(vertex_pos) = data.vertex_pos {
                        Some(dc::octree::LeafNode::new(
                            rel_pos,
                            node.size(),
                            dc::contour::NodeData {
                                corners: data.corners,
                                vertex_pos,
                                vertex_index: 0,
                                is_seam: true,
                                data: data.data,
                            },
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            nodes.extend(neighbour_nodes);
        }

        let mut vertices = Vec::with_capacity(nodes.len());

        // Reindex nodes vertex indices
        for node in &mut nodes {
            {
                let node_data = node.data_mut();
                node_data.vertex_index = vertices.len() as u32;
            }

            let node_pos = node.position();
            let node_pos = na::Vector3::new(node_pos.x as f32, node_pos.y as f32, node_pos.z as f32);

            let pos = node_pos + node.data().vertex_pos * (node.size() as f32);
            let data = &node.data().data;

            let material_ids = na::Vector4::new(
                data[0].material as u32 | ((data[1].material as u32) << 16),
                data[2].material as u32 | ((data[3].material as u32) << 16),
                data[4].material as u32 | ((data[5].material as u32) << 16),
                data[6].material as u32 | ((data[7].material as u32) << 16),
            );

            vertices.push(Vertex {
                position: pos,
                normal: Default::default(),
                material_ids,
            });
        }

        // Create octree & generate mesh
        let octree = dc::octree::from_nodes((SECTOR_SIZE * 2) as u32 * self.node_size, &nodes);
        let indices = dc::contour::generate_mesh(&octree);

        (vertices, indices)
    }

    // Collect nodes to use in seams of another clusters
    pub fn collect_nodes_for_seams(
        &mut self,
    ) -> &HashMap<u32, Vec<dc::octree::LeafNode<dc::contour::NodeDataDiscrete<PointData>>>> {
        self.update_seams();

        let layer_count = self.layer_count();

        for layer_index in 0..layer_count {
            if let hash_map::Entry::Vacant(entry) = self.seam_nodes_cache.entry(layer_index) {
                let mut nodes = Vec::with_capacity(SIZE * SIZE * 3);
                let mut temp_densities = [0.0_f32; 8];
                let mut temp_mats = [PointData::default(); 8];

                for xs in 0..SIZE_IN_SECTORS {
                    for ys in 0..SIZE_IN_SECTORS {
                        for zs in 0..SIZE_IN_SECTORS {
                            let sector = &self.sectors[xs][ys][zs];
                            let density_indices = &sector.indices;
                            let densities = &sector.densities;

                            macro_rules! collect_density {
                                ($x: expr, $y: expr, $z: expr) => {
                                    let mut indices = [
                                        density_indices[$x][$y][$z] as usize,
                                        density_indices[$x][$y][$z + 1] as usize,
                                        density_indices[$x][$y + 1][$z] as usize,
                                        density_indices[$x][$y + 1][$z + 1] as usize,
                                        density_indices[$x + 1][$y][$z] as usize,
                                        density_indices[$x + 1][$y][$z + 1] as usize,
                                        density_indices[$x + 1][$y + 1][$z] as usize,
                                        density_indices[$x + 1][$y + 1][$z + 1] as usize,
                                    ];

                                    let mut is_valid_cell = true;

                                    for i in 0..8 {
                                        let count = indices[i] >> 24;
                                        indices[i] &= 0x00ffffff;

                                        if layer_index >= count as u32 {
                                            is_valid_cell = false;
                                            break;
                                        }
                                    }

                                    let pos = na::Vector3::new(
                                        (xs * SECTOR_SIZE + $x) as u32 * self.node_size,
                                        (ys * SECTOR_SIZE + $y) as u32 * self.node_size,
                                        (zs * SECTOR_SIZE + $z) as u32 * self.node_size,
                                    );

                                    if !is_valid_cell {
                                        continue;
                                    }

                                    for i in 0..8 {
                                        let p = densities[indices[i] + layer_index as usize];
                                        temp_densities[i] = (p.density as f32) / 255.0;
                                        temp_mats[i] = PointData {
                                            material: p.material,
                                        };
                                    }

                                    let node_data = dc::contour::NodeDataDiscrete::new(
                                        temp_densities,
                                        ISO_VALUE_NORM,
                                        temp_mats,
                                    );
                                    nodes.push(dc::octree::LeafNode::new(pos, self.node_size, node_data));
                                };
                            }

                            if zs == 0 {
                                for i in 0..SECTOR_SIZE {
                                    for j in 0..SECTOR_SIZE {
                                        collect_density!(i, j, 0);
                                    }
                                }
                            }
                            if ys == 0 {
                                for i in 0..SECTOR_SIZE {
                                    for j in 0..SECTOR_SIZE {
                                        if zs != 0 || j != 0 {
                                            collect_density!(i, 0, j);
                                        }
                                    }
                                }
                            }
                            if xs == 0 {
                                for i in 0..SECTOR_SIZE {
                                    for j in 0..SECTOR_SIZE {
                                        if (zs != 0 || j != 0) && (ys != 0 || i != 0) {
                                            collect_density!(0, i, j);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                entry.insert(nodes);
            }
        }

        &self.seam_nodes_cache
    }

    pub fn update_mesh(&mut self, seam: &Seam, simplification_factor: f32) {
        if seam.node_size != self.node_size {
            panic!(
                "seam.node_size ({}) != self.node_size ({})",
                seam.node_size, self.node_size
            );
        }

        // Collect vertices & indices
        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    let sector_changed = self.sectors[x][y][z].changed;

                    if sector_changed {
                        let sector = &self.sectors[x][y][z];
                        let mut sector_vertices = Vec::with_capacity(SECTOR_VOLUME * MAX_CELL_LAYERS);
                        let mut sector_indices = Vec::with_capacity(SECTOR_VOLUME * MAX_CELL_LAYERS);

                        for i in 0..sector.layer_count {
                            let def_seam_layer = SeamLayer::default();

                            let (mut temp_vertices, temp_indices) = self.triangulate(
                                [x as u32, y as u32, z as u32],
                                i,
                                seam.nodes.get(&i).unwrap_or(&def_seam_layer),
                            );

                            let (temp_vertices, temp_indices) = {
                                let options = mesh_simplifier::Options::new(
                                    0.125,
                                    10,
                                    (512 as f32 * (1.0 - simplification_factor)) as usize,
                                    4.0 * self.node_size as f32,
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
                        }

                        let sector = &mut self.sectors[x][y][z];
                        sector.vertex_count = sector_vertices.len() as u32;
                        sector.index_count = sector_indices.len() as u32;

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
                let node_size = self.node_size as usize;

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
        node_size,
        seam_nodes_cache: Default::default(),
        nodes_buffer: None,
        device: Arc::clone(device),
    }
}
