use crate::renderer::vertex_mesh::{VertexMesh, VertexMeshCreate};
use crate::utils;
use dual_contouring as dc;
use nalgebra as na;
use std::convert::TryInto;
use std::sync::Arc;
use vk_wrapper as vkw;

const SECTOR_SIZE: usize = 16;
const ALIGNED_SECTOR_SIZE: usize = SECTOR_SIZE + 2;
const SIZE_IN_SECTORS: usize = 4;
pub const SIZE: usize = SECTOR_SIZE * SIZE_IN_SECTORS;
const ALIGNED_SIZE: usize = SIZE + 2;
pub const MAX_CELL_LAYERS: usize = 4; // MAX: 255
const SECTOR_VOLUME: usize = SECTOR_SIZE * SECTOR_SIZE * SECTOR_SIZE;
const ALIGNED_SECTOR_VOLUME: usize = ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE;
const ALIGNED_SECTOR_MAX_CELLS: usize = ALIGNED_SECTOR_VOLUME * MAX_CELL_LAYERS;
const ISO_VALUE_NORM: f32 = 0.5;
const ISO_VALUE_INT: i16 = (ISO_VALUE_NORM * 255.0) as i16;

pub fn index_3d_to_1d(p: [u8; 3], ds: u32) -> u32 {
    (p[2] as u32) + (p[1] as u32) * ds + (p[0] as u32) * ds * ds
}

/*macro_rules! index_3d_to_1d {
    ($p: expr, $ds: expr) => {
        ($p[2] as u32) + ($p[1] as u32) * $ds + ($p[0] as u32) * $ds * $ds
    };
}*/

macro_rules! index_3d_to_1d_inv {
    ($p: expr, $ds: expr) => {
        ($p[0] as u32) + ($p[1] as u32) * $ds + ($p[2] as u32) * $ds * $ds
    };
}

pub fn calc_density_index(head_index: u32, layer_count: u8) -> u32 {
    head_index | ((layer_count as u32) << 24)
}

#[derive(Debug, Default, Copy, Clone)]
pub struct DensityPoint {
    pub(crate) density: u8,
    pub(crate) material: u16,
}

#[derive(Clone)]
pub struct Sector {
    indices: Box<[[[u32; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]>,
    densities: Vec<DensityPoint>,
    indices_changed: bool,
    changed: bool,
    vertices_offset: u32,
    vertex_count: u32,
    indices_offset: u32,
    index_count: u32,
}

#[derive(Copy, Clone)]
pub struct DensityPointInfo {
    /// [x, y, z, layer index]
    pub(crate) pos: [u8; 4],
    pub(crate) point: DensityPoint,
}

impl Sector {
    /// pos: [x, y, z, layer index]
    pub fn get_density(&self, pos: [u8; 4]) -> Option<DensityPoint> {
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
    pub fn get_density_layers(&self, pos: [u8; 3], out: &mut [DensityPoint]) -> u8 {
        let index = &self.indices[pos[0] as usize][pos[1] as usize][pos[2] as usize];
        let head_index = index & 0x00ffffff;
        let layer_count = (index & 0xff000000) >> 24;
        let read_count = out.len().min(layer_count as usize);

        for i in 0..read_count {
            out[i] = self.densities[head_index as usize + i];
        }

        return read_count as u8;
    }

    /// pos: [x, y, z, layer index]
    pub fn set_density(&mut self, pos: [u8; 4], density: DensityPoint) {
        let index = &mut self.indices[pos[0] as usize][pos[1] as usize][pos[2] as usize];
        let head_index = *index & 0x00ffffff;
        let layer_count = (*index & 0xff000000) >> 24;

        let insert_index = head_index as usize + pos[3] as usize;

        if pos[3] < MAX_CELL_LAYERS as u8 {
            if pos[3] < layer_count as u8 {
                self.densities[insert_index] = density;
            } else if pos[3] == layer_count as u8 {
                self.densities.insert(insert_index, density);
                *index = head_index | ((layer_count + 1) << 24);
                self.indices_changed = true;
                self.update_indices();
            }
        }
    }

    fn set_densities(&mut self, points: &mut [DensityPointInfo]) {
        // Sort points by position starting from [0,,] to [sector_SIZE,,]
        points.sort_by(|a, b| {
            let a_dist = index_3d_to_1d((&a.pos[..3]).try_into().unwrap(), ALIGNED_SECTOR_SIZE as u32);
            let b_dist = index_3d_to_1d((&b.pos[..3]).try_into().unwrap(), ALIGNED_SECTOR_SIZE as u32);

            a_dist.cmp(&b_dist)
        });

        let mut offset = 0u32;

        for point_info in points {
            let pos = [
                point_info.pos[0] as usize,
                point_info.pos[1] as usize,
                point_info.pos[2] as usize,
            ];

            let index = &mut self.indices[pos[0]][pos[1]][pos[2]];
            let head_index = *index & 0x00ffffff;
            let layer_count = (*index & 0xff000000) >> 24;

            let insert_index = (head_index + point_info.pos[3] as u32 + offset) as usize;

            if point_info.pos[3] < MAX_CELL_LAYERS as u8 {
                if point_info.pos[3] < layer_count as u8 {
                    self.densities[insert_index] = point_info.point;
                } else if point_info.pos[3] == layer_count as u8 {
                    // Add a new layer
                    self.densities.insert(insert_index, point_info.point);
                    *index = head_index | ((layer_count + 1) << 24);
                    offset += 1;
                    self.indices_changed = true;
                }
            }
        }

        self.update_indices();
    }

    pub fn update_indices(&mut self) {
        if !self.indices_changed {
            return;
        }
        self.indices_changed = false;
        self.changed = true;

        let mut offset = 0u32;

        for x in 0..ALIGNED_SECTOR_SIZE {
            for y in 0..ALIGNED_SECTOR_SIZE {
                for z in 0..ALIGNED_SECTOR_SIZE {
                    let index = &mut self.indices[x][y][z];
                    let layer_count = (*index & 0xff000000) >> 24;

                    *index = offset | (layer_count << 24);
                    offset += layer_count;
                }
            }
        }
    }
}

impl Default for Sector {
    fn default() -> Self {
        Self {
            indices: Box::new([[[0; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]; ALIGNED_SECTOR_SIZE]),
            densities: vec![],
            indices_changed: false,
            changed: false,
            vertices_offset: 0,
            vertex_count: 0,
            indices_offset: 0,
            index_count: 0,
        }
    }
}

pub struct Cluster {
    sectors: [[[Sector; SIZE_IN_SECTORS]; SIZE_IN_SECTORS]; SIZE_IN_SECTORS],
    vertex_mesh: VertexMesh<Vertex>,
}

#[derive(Copy, Clone, Default)]
pub struct Vertex {
    position: na::Vector3<f32>,
    density: u32,
}
vertex_impl!(Vertex, position, density);

impl Cluster {
    fn calc_cell_point(points: &[DensityPoint; 8]) -> Option<na::Vector3<f32>> {
        let mut c_count = 0u32;
        let mut vertex = na::Vector3::<f32>::new(0.0, 0.0, 0.0);

        fn cell_interpolate(a: i16, b: i16) -> f32 {
            ((ISO_VALUE_INT - a) as f32) / ((b - a) as f32)
        }

        for x in 0..2 {
            for y in 0..2 {
                let i0 = index_3d_to_1d_inv!([x, y, 0], 2) as usize;
                let i1 = index_3d_to_1d_inv!([x, y, 1], 2) as usize;

                if (points[i0].density > ISO_VALUE_INT as u8) != (points[i1].density > ISO_VALUE_INT as u8) {
                    vertex += na::Vector3::new(
                        x as f32,
                        y as f32,
                        cell_interpolate(points[i0].density as i16, points[i1].density as i16) as f32,
                    );
                    c_count += 1;
                }
            }
        }

        for x in 0..2 {
            for z in 0..2 {
                let i0 = index_3d_to_1d_inv!([x, 0, z], 2) as usize;
                let i1 = index_3d_to_1d_inv!([x, 1, z], 2) as usize;

                if (points[i0].density > ISO_VALUE_INT as u8) != (points[i1].density > ISO_VALUE_INT as u8) {
                    vertex += na::Vector3::new(
                        x as f32,
                        cell_interpolate(points[i0].density as i16, points[i1].density as i16) as f32,
                        z as f32,
                    );
                    c_count += 1;
                }
            }
        }

        for y in 0..2 {
            for z in 0..2 {
                let i0 = index_3d_to_1d_inv!([0, y, z], 2) as usize;
                let i1 = index_3d_to_1d_inv!([1, y, z], 2) as usize;

                if (points[i0].density > ISO_VALUE_INT as u8) != (points[i1].density > ISO_VALUE_INT as u8) {
                    vertex += na::Vector3::new(
                        cell_interpolate(points[i0].density as i16, points[i1].density as i16) as f32,
                        y as f32,
                        z as f32,
                    );
                    c_count += 1;
                }
            }
        }

        if c_count < 2 {
            None
        } else {
            Some(vertex / (c_count as f32))
        }
    }

    pub fn calc_sector_position(cell_pos: [u8; 3]) -> [usize; 3] {
        [
            (cell_pos[0] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
            (cell_pos[1] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
            (cell_pos[2] as usize / SECTOR_SIZE).min(SIZE_IN_SECTORS - 1),
        ]
    }

    pub fn vertex_mesh(&self) -> &VertexMesh<Vertex> {
        &self.vertex_mesh
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
        sector.get_density_layers(pos, out)
    }

    /// pos: [x, y, z, layer index]
    pub fn set_density(&mut self, pos: [u8; 4], density: DensityPoint) {
        let sector_pos = Self::calc_sector_position((&pos[..3]).try_into().unwrap());
        let sector = &mut self.sectors[sector_pos[0]][sector_pos[1]][sector_pos[2]];

        sector.set_density(
            [
                pos[0] % SECTOR_SIZE as u8,
                pos[1] % SECTOR_SIZE as u8,
                pos[2] % SECTOR_SIZE as u8,
                pos[3],
            ],
            density,
        );
    }

    pub fn set_densities(&mut self, points: &[DensityPointInfo]) {
        // Sort points by position starting from [0,,] to [SIZE,,]
        let mut points = points.to_vec();
        points.sort_by(|a, b| {
            let a_dist = index_3d_to_1d((&a.pos[..3]).try_into().unwrap(), ALIGNED_SIZE as u32);
            let b_dist = index_3d_to_1d((&b.pos[..3]).try_into().unwrap(), ALIGNED_SIZE as u32);

            a_dist.cmp(&b_dist)
        });

        // Offset for each sector for density insert (density_indices isn't updated every insertion to densities)
        let mut offset = [[[0u32; SIZE_IN_SECTORS]; SIZE_IN_SECTORS]; SIZE_IN_SECTORS];

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

            let sector_offset = &mut offset[sector_pos[0]][sector_pos[1]][sector_pos[2]];
            let insert_index = (head_index + point_info.pos[3] as u32 + *sector_offset) as usize;

            if point_info.pos[3] < MAX_CELL_LAYERS as u8 {
                if point_info.pos[3] < layer_count as u8 {
                    sector.densities[insert_index] = point_info.point;
                } else if point_info.pos[3] == layer_count as u8 {
                    // Add a new layer
                    sector.densities.insert(insert_index, point_info.point);
                    *index = head_index | ((layer_count + 1) << 24);
                    *sector_offset += 1;
                    sector.indices_changed = true;
                }
            }
        }

        // Update indices to account for inserted densities
        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    self.sectors[x][y][z].update_indices();
                }
            }
        }
    }

    // TODO: sparse update (cache various buffers into Cluster struct)
    /// details: max detail deviation [0;1]
    fn triangulate(&mut self, sector_pos: [u8; 3], details: f32) -> (Vec<Vertex>, Vec<u32>) {
        // Fill sectors edges (use sectors or self.adjacency)
        {
            let mut temp_density_infos =
                Vec::<DensityPointInfo>::with_capacity(SECTOR_SIZE * SECTOR_SIZE * 8);
            let mut temp_density = [DensityPoint::default(); MAX_CELL_LAYERS];

            // Right side
            if sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8 {
                let sector =
                    &self.sectors[sector_pos[0] as usize + 1][sector_pos[1] as usize][sector_pos[2] as usize];

                for x in 0..2 {
                    for y in 0..SECTOR_SIZE {
                        for z in 0..SECTOR_SIZE {
                            let count =
                                sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

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
                            let count =
                                sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

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
                            let count =
                                sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

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
            if (sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8) && (sector_pos[1] < (SIZE_IN_SECTORS - 1) as u8)
            {
                let sector = &self.sectors[sector_pos[0] as usize + 1][sector_pos[1] as usize + 1]
                    [sector_pos[2] as usize];

                for x in 0..2 {
                    for y in 0..2 {
                        for z in 0..SECTOR_SIZE {
                            let count =
                                sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

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
            if (sector_pos[0] < (SIZE_IN_SECTORS - 1) as u8) && (sector_pos[2] < (SIZE_IN_SECTORS - 1) as u8)
            {
                let sector = &self.sectors[sector_pos[0] as usize + 1][sector_pos[1] as usize]
                    [sector_pos[2] as usize + 1];

                for x in 0..2 {
                    for y in 0..SECTOR_SIZE {
                        for z in 0..2 {
                            let count =
                                sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

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
            if (sector_pos[1] < (SIZE_IN_SECTORS - 1) as u8) && (sector_pos[2] < (SIZE_IN_SECTORS - 1) as u8)
            {
                let sector = &self.sectors[sector_pos[0] as usize][sector_pos[1] as usize + 1]
                    [sector_pos[2] as usize + 1];

                for x in 0..SECTOR_SIZE {
                    for y in 0..2 {
                        for z in 0..2 {
                            let count =
                                sector.get_density_layers([x as u8, y as u8, z as u8], &mut temp_density);

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

        let sector = &self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize];
        let densities = &sector.densities;
        let density_indices = &sector.indices;

        // Vertex components
        let mut v_positions: Vec<na::Vector3<f32>> =
            vec![na::Vector3::new(f32::NAN, f32::NAN, f32::NAN); ALIGNED_SECTOR_MAX_CELLS];
        let mut v_density_indices = vec![0u32; ALIGNED_SECTOR_MAX_CELLS];

        let mut v_faces2: Vec<Option<([u32; 4], bool)>> = vec![None; ALIGNED_SECTOR_MAX_CELLS * 3];
        let mut v_face_normals = vec![None; ALIGNED_SECTOR_MAX_CELLS * 3];

        // Faces index buffer
        let mut v_faces = Vec::<[u32; 4]>::with_capacity(ALIGNED_SECTOR_MAX_CELLS * 3);
        // Density buffer
        let mut density_infos = Vec::<[u32; 4]>::with_capacity(ALIGNED_SECTOR_VOLUME);

        macro_rules! calc_index {
            ($x: expr, $y: expr, $z: expr) => {
                (($z + $y * ALIGNED_SECTOR_SIZE + $x * ALIGNED_SECTOR_SIZE * ALIGNED_SECTOR_SIZE)
                    * MAX_CELL_LAYERS) as u32
            };
        }

        for x in 0..(SECTOR_SIZE + 1) {
            for y in 0..(SECTOR_SIZE + 1) {
                for z in 0..(SECTOR_SIZE + 1) {
                    // Obtain 2x2x2 indices for v_positions & v_density_indices arrays
                    let mut indices = [
                        density_indices[x][y][z],
                        density_indices[x + 1][y][z],
                        density_indices[x][y + 1][z],
                        density_indices[x + 1][y + 1][z],
                        density_indices[x][y][z + 1],
                        density_indices[x + 1][y][z + 1],
                        density_indices[x][y + 1][z + 1],
                        density_indices[x + 1][y + 1][z + 1],
                    ];
                    let arr_indices = [
                        calc_index!(x, y, z),
                        calc_index!(x + 1, y, z),
                        calc_index!(x, y + 1, z),
                        calc_index!(x + 1, y + 1, z),
                        calc_index!(x, y, z + 1),
                        calc_index!(x + 1, y, z + 1),
                        calc_index!(x, y + 1, z + 1),
                        calc_index!(x + 1, y + 1, z + 1),
                    ];

                    // Get layer counts
                    let mut layer_counts = [0u32; 8];

                    // Calculate max layer count & validate cell
                    let mut max_count = 0;

                    let mut is_valid_cell = true;
                    for i in 0..8 {
                        layer_counts[i] = indices[i] >> 24;
                        indices[i] &= 0x00ffffff;
                        max_count = max_count.max(layer_counts[i]);

                        if layer_counts[i] == 0 {
                            is_valid_cell = false;
                            break;
                        }
                    }

                    if !is_valid_cell {
                        continue;
                    }

                    for i in 0..max_count {
                        // Get points of current layer of current cell
                        let points = [
                            densities[(indices[0] + (layer_counts[0] - 1).min(i)) as usize],
                            densities[(indices[1] + (layer_counts[1] - 1).min(i)) as usize],
                            densities[(indices[2] + (layer_counts[2] - 1).min(i)) as usize],
                            densities[(indices[3] + (layer_counts[3] - 1).min(i)) as usize],
                            densities[(indices[4] + (layer_counts[4] - 1).min(i)) as usize],
                            densities[(indices[5] + (layer_counts[5] - 1).min(i)) as usize],
                            densities[(indices[6] + (layer_counts[6] - 1).min(i)) as usize],
                            densities[(indices[7] + (layer_counts[7] - 1).min(i)) as usize],
                        ];

                        // Store cell material info
                        let density_index = density_infos.len();
                        density_infos.push([
                            ((points[0].material as u32) << 16) | (points[1].material as u32),
                            ((points[2].material as u32) << 16) | (points[3].material as u32),
                            ((points[4].material as u32) << 16) | (points[5].material as u32),
                            ((points[6].material as u32) << 16) | (points[7].material as u32),
                        ]);

                        // Calculate point in the cell
                        if let Some(point) = Self::calc_cell_point(&points) {
                            let point = point + na::Vector3::new(x as f32, y as f32, z as f32)
                                - na::Vector3::new(1.0, 1.0, 1.0);
                            let index = (arr_indices[0] + i.min(layer_counts[0] - 1) as u32) as usize;

                            v_positions[index] = point;
                            v_density_indices[index] = density_index as u32;
                        }

                        // Generate quads
                        // ---------------------------------------------------------------------------------------------
                        // XY plane
                        let s1 = points[3].density > ISO_VALUE_INT as u8;
                        let s2 = points[7].density > ISO_VALUE_INT as u8;
                        if s1 != s2 {
                            if s1 {
                                v_faces2[(arr_indices[0] as usize + i as usize) * 3] = Some((
                                    [
                                        arr_indices[2] + i.min(layer_counts[2] - 1),
                                        arr_indices[3] + i.min(layer_counts[3] - 1),
                                        arr_indices[1] + i.min(layer_counts[1] - 1),
                                        arr_indices[0] + i.min(layer_counts[0] - 1),
                                    ],
                                    false,
                                ));
                            } else {
                                v_faces2[(arr_indices[0] as usize + i as usize) * 3] = Some((
                                    [
                                        arr_indices[2] + i.min(layer_counts[2] - 1),
                                        arr_indices[3] + i.min(layer_counts[3] - 1),
                                        arr_indices[1] + i.min(layer_counts[1] - 1),
                                        arr_indices[0] + i.min(layer_counts[0] - 1),
                                        /*indices[0] + i.min(densities[0].1 - 1) as u32,
                                        indices[1] + i.min(densities[1].1 - 1) as u32,
                                        indices[3] + i.min(densities[3].1 - 1) as u32,
                                        indices[2] + i.min(densities[2].1 - 1) as u32,*/
                                    ],
                                    true,
                                ));
                            }
                        }

                        // XZ plane
                        let s1 = points[5].density > ISO_VALUE_INT as u8;
                        if s1 != s2 {
                            if s2 {
                                v_faces2[(arr_indices[0] as usize + i as usize) * 3 + 1] = Some((
                                    [
                                        arr_indices[4] + i.min(layer_counts[4] - 1),
                                        arr_indices[5] + i.min(layer_counts[5] - 1),
                                        arr_indices[1] + i.min(layer_counts[1] - 1),
                                        arr_indices[0] + i.min(layer_counts[0] - 1),
                                    ],
                                    false,
                                ));
                            } else {
                                v_faces2[(arr_indices[0] as usize + i as usize) * 3 + 1] = Some((
                                    [
                                        arr_indices[4] + i.min(layer_counts[4] - 1),
                                        arr_indices[5] + i.min(layer_counts[5] - 1),
                                        arr_indices[1] + i.min(layer_counts[1] - 1),
                                        arr_indices[0] + i.min(layer_counts[0] - 1),
                                        /*indices[0] + i.min(densities[0].1 - 1) as u32,
                                        indices[1] + i.min(densities[1].1 - 1) as u32,
                                        indices[5] + i.min(densities[5].1 - 1) as u32,
                                        indices[4] + i.min(densities[4].1 - 1) as u32,*/
                                    ],
                                    true,
                                ));
                            }
                        }

                        // YZ plane
                        let s1 = points[6].density > ISO_VALUE_INT as u8;
                        if s1 != s2 {
                            if s1 {
                                v_faces2[(arr_indices[0] as usize + i as usize) * 3 + 2] = Some((
                                    [
                                        arr_indices[4] + i.min(layer_counts[4] - 1),
                                        arr_indices[6] + i.min(layer_counts[6] - 1),
                                        arr_indices[2] + i.min(layer_counts[2] - 1),
                                        arr_indices[0] + i.min(layer_counts[0] - 1),
                                    ],
                                    false,
                                ));
                            } else {
                                v_faces2[(arr_indices[0] as usize + i as usize) * 3 + 2] = Some((
                                    [
                                        arr_indices[4] + i.min(layer_counts[4] - 1),
                                        arr_indices[6] + i.min(layer_counts[6] - 1),
                                        arr_indices[2] + i.min(layer_counts[2] - 1),
                                        arr_indices[0] + i.min(layer_counts[0] - 1),
                                        /*indices[0] + i.min(densities[0].1 - 1) as u32,
                                        indices[2] + i.min(densities[2].1 - 1) as u32,
                                        indices[6] + i.min(densities[6].1 - 1) as u32,
                                        indices[4] + i.min(densities[4].1 - 1) as u32,*/
                                    ],
                                    true,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Optimize mesh
        // -------------------------------------------------------------------------------------------------------------
        let mut use_marks = vec![false; ALIGNED_SECTOR_MAX_CELLS * 3];
        let mut merge_marks = vec![false; ALIGNED_SECTOR_MAX_CELLS];
        //if details > 0.0 {

        fn calc_normal(vertices: [na::Vector3<f32>; 4]) -> na::Vector3<f32> {
            let normal0 = utils::calc_triangle_normal(&vertices[0], &vertices[1], &vertices[2]);
            let normal1 = utils::calc_triangle_normal(&vertices[0], &vertices[2], &vertices[3]);
            (normal0 + normal1) / 2.0
        }
        macro_rules! get_normal {
            ($face: expr, $index: expr) => {{
                let normal = &mut v_face_normals[$index as usize];

                if normal.is_none() {
                    let v0 = v_positions[$face.0[0] as usize];
                    let v1 = v_positions[$face.0[1] as usize];
                    let v2 = v_positions[$face.0[2] as usize];
                    let v3 = v_positions[$face.0[3] as usize];

                    *normal = Some(calc_normal([v0, v1, v2, v3]));
                }

                normal.unwrap()
            }};
        }
        macro_rules! check_merge {
            ($x: expr, $y: expr, $z: expr, $layer: expr, $s: expr, $normal: expr) => {{
                let cell_index = calc_index!($x, $y, $z);
                let index = (cell_index as usize + $layer as usize) * 3 + $s;
                let face = v_faces2[index];

                if let Some(face) = face {
                    let normal2 = get_normal!(&face, index);
                    let diff = 1.0 - $normal.dot(&normal2);
                    diff < details
                } else {
                    false
                }
            }};
        }

        for x in 0..SECTOR_SIZE {
            for y in 0..SECTOR_SIZE {
                for z in 0..SECTOR_SIZE {
                    let cell_index = calc_index!(x, y, z);

                    if x > 0
                        && y > 0
                        && z > 0
                        && x < SECTOR_SIZE - 1
                        && y < SECTOR_SIZE - 1
                        && z < SECTOR_SIZE - 1
                    {
                        for i in 0..MAX_CELL_LAYERS {
                            // XY plane
                            {
                                let check_quad_usage = |x: usize, y: usize| -> bool {
                                    use_marks[(calc_index!(x, y, z) as usize + i) * 3 + 0]
                                };
                                let check_merge_marks = |x: usize, y: usize| -> bool {
                                    merge_marks[calc_index!(x, y, z) as usize + i]
                                        || merge_marks[calc_index!(x + 1, y, z) as usize + i]
                                        || merge_marks[calc_index!(x, y + 1, z) as usize + i]
                                        || merge_marks[calc_index!(x + 1, y + 1, z) as usize + i]
                                };

                                let index = (cell_index as usize + i) * 3 + 0;
                                let face = v_faces2[index];

                                if !check_quad_usage(x, y) {
                                    if let Some(face) = face {
                                        let normal = get_normal!(&face, index);

                                        let mut w = 1;
                                        let mut h = 1;

                                        if !check_merge_marks(x, y) {
                                            let mut prev_max_y2 = SECTOR_SIZE - y - 2;

                                            for x2 in 0..(SECTOR_SIZE - x - 2) {
                                                let mut max_y2 = 0;

                                                for y2 in 0..prev_max_y2 {
                                                    if check_quad_usage(x + x2, y + y2)
                                                        || check_merge_marks(x + x2, y + y2)
                                                        || !check_merge!(x + x2, y + y2, z, i, 0, normal)
                                                    {
                                                        break;
                                                    }

                                                    max_y2 += 1;
                                                }

                                                if (x2 * prev_max_y2) < ((x2 + 1) * max_y2) {
                                                    w = x2 + 1;
                                                    h = max_y2;
                                                    prev_max_y2 = max_y2;
                                                } else {
                                                    break;
                                                }
                                            }
                                        }

                                        // Mark used quads
                                        if w == 1 && h == 1 {
                                            let index = (calc_index!(x, y, z) as usize + i) * 3 + 0;
                                            use_marks[index] = true;
                                        } else {
                                            for x2 in 0..(w + 1) {
                                                for y2 in 0..(h + 1) {
                                                    let index = calc_index!(x + x2, y + y2, z) as usize + i;
                                                    merge_marks[index] = true;

                                                    if x2 < w && y2 < h {
                                                        use_marks[(index) * 3 + 0] = true;
                                                    }
                                                }
                                            }
                                        }

                                        // Create new quad
                                        let face_x0_y0 = &face;
                                        let face_x1_y0 = v_faces2
                                            [(calc_index!(x + w - 1, y, z) as usize + i) * 3 + 0]
                                            .as_ref()
                                            .unwrap();
                                        let face_x0_y1 = v_faces2
                                            [(calc_index!(x, y + h - 1, z) as usize + i) * 3 + 0]
                                            .as_ref()
                                            .unwrap();
                                        let face_x1_y1 = v_faces2
                                            [(calc_index!(x + w - 1, y + h - 1, z) as usize + i) * 3 + 0]
                                            .as_ref()
                                            .unwrap();

                                        if face.1 {
                                            v_faces.push([
                                                face_x0_y0.0[3],
                                                face_x1_y0.0[2],
                                                face_x1_y1.0[1],
                                                face_x0_y1.0[0],
                                            ]);
                                        } else {
                                            v_faces.push([
                                                face_x0_y1.0[0],
                                                face_x1_y1.0[1],
                                                face_x1_y0.0[2],
                                                face_x0_y0.0[3],
                                            ]);
                                        }

                                        // Set correct boundary vertices
                                        let x0_start = v_positions[face_x0_y0.0[3] as usize];
                                        let x0_end = v_positions[face_x0_y1.0[0] as usize];
                                        let x0_step = (x0_end - x0_start) / (h as f32);

                                        let x1_start = v_positions[face_x1_y0.0[2] as usize];
                                        let x1_end = v_positions[face_x1_y1.0[1] as usize];
                                        let x1_step = (x1_end - x1_start) / (h as f32);

                                        let y0_start = v_positions[face_x0_y0.0[3] as usize];
                                        let y0_end = v_positions[face_x1_y0.0[2] as usize];
                                        let y0_step = (y0_end - y0_start) / (w as f32);

                                        let y1_start = v_positions[face_x0_y1.0[0] as usize];
                                        let y1_end = v_positions[face_x1_y1.0[1] as usize];
                                        let y1_step = (y1_end - y1_start) / (w as f32);

                                        for x2 in 1..w {
                                            // y0 edge
                                            let index = calc_index!(x + x2, y, z) as usize + i;
                                            v_positions[index] = y0_start + y0_step * (x2 as f32);

                                            // y1 edge
                                            let index = calc_index!(x + x2, y + h, z) as usize + i;
                                            v_positions[index] = y1_start + y1_step * (x2 as f32);
                                        }

                                        for y2 in 1..h {
                                            // x0 edge
                                            let index = calc_index!(x, y + y2, z) as usize + i;
                                            v_positions[index] = x0_start + x0_step * (y2 as f32);

                                            // x1 edge
                                            let index = calc_index!(x + w, y + y2, z) as usize + i;
                                            v_positions[index] = x1_start + x1_step * (y2 as f32);
                                        }
                                    }
                                }
                            }

                            // XZ plane
                            {
                                let check_quad_usage = |x: usize, z: usize| -> bool {
                                    use_marks[(calc_index!(x, y, z) as usize + i) * 3 + 1]
                                };
                                let check_merge_marks = |x: usize, z: usize| -> bool {
                                    merge_marks[calc_index!(x, y, z) as usize + i]
                                        || merge_marks[calc_index!(x + 1, y, z) as usize + i]
                                        || merge_marks[calc_index!(x, y, z + 1) as usize + i]
                                        || merge_marks[calc_index!(x + 1, y, z + 1) as usize + i]
                                };

                                let index = (cell_index as usize + i) * 3 + 1;
                                let face = v_faces2[index];

                                if !check_quad_usage(x, z) {
                                    if let Some(face) = face {
                                        let normal = get_normal!(&face, index);

                                        let mut w = 1;
                                        let mut h = 1;

                                        if !check_merge_marks(x, z) {
                                            let mut prev_max_z2 = SECTOR_SIZE - z - 2;

                                            for x2 in 0..(SECTOR_SIZE - x - 2) {
                                                let mut max_z2 = 0;

                                                for z2 in 0..prev_max_z2 {
                                                    if check_quad_usage(x + x2, z + z2)
                                                        || check_merge_marks(x + x2, z + z2)
                                                        || !check_merge!(x + x2, y, z + z2, i, 1, normal)
                                                    {
                                                        break;
                                                    }

                                                    max_z2 += 1;
                                                }

                                                if (x2 * prev_max_z2) < ((x2 + 1) * max_z2) {
                                                    w = x2 + 1;
                                                    h = max_z2;
                                                    prev_max_z2 = max_z2;
                                                } else {
                                                    break;
                                                }
                                            }
                                        }

                                        // Mark used quads
                                        if w == 1 && h == 1 {
                                            let index = (calc_index!(x, y, z) as usize + i) * 3 + 1;
                                            use_marks[index] = true;
                                        } else {
                                            for x2 in 0..(w + 1) {
                                                for z2 in 0..(h + 1) {
                                                    let index = calc_index!(x + x2, y, z + z2) as usize + i;
                                                    merge_marks[index] = true;

                                                    if x2 < w && z2 < h {
                                                        use_marks[(index) * 3 + 1] = true;
                                                    }
                                                }
                                            }
                                        }

                                        // Create new quad
                                        let face_x0_z0 = &face;
                                        let face_x1_z0 = v_faces2
                                            [(calc_index!(x + w - 1, y, z) as usize + i) * 3 + 1]
                                            .as_ref()
                                            .unwrap();
                                        let face_x0_z1 = v_faces2
                                            [(calc_index!(x, y, z + h - 1) as usize + i) * 3 + 1]
                                            .as_ref()
                                            .unwrap();
                                        let face_x1_z1 = v_faces2
                                            [(calc_index!(x + w - 1, y, z + h - 1) as usize + i) * 3 + 1]
                                            .as_ref()
                                            .unwrap();

                                        /*v_faces.push([
                                            face_x0_z1[0],
                                            face_x1_z1[1],
                                            face_x1_z0[2],
                                            face_x0_z0[3],
                                        ]);*/

                                        if face.1 {
                                            v_faces.push([
                                                face_x0_z0.0[3],
                                                face_x1_z0.0[2],
                                                face_x1_z1.0[1],
                                                face_x0_z1.0[0],
                                            ]);
                                        } else {
                                            v_faces.push([
                                                face_x0_z1.0[0],
                                                face_x1_z1.0[1],
                                                face_x1_z0.0[2],
                                                face_x0_z0.0[3],
                                            ]);
                                        }

                                        // Set correct boundary vertices
                                        let x0_start = v_positions[face_x0_z0.0[3] as usize];
                                        let x0_end = v_positions[face_x0_z1.0[0] as usize];
                                        let x0_step = (x0_end - x0_start) / (h as f32);

                                        let x1_start = v_positions[face_x1_z0.0[2] as usize];
                                        let x1_end = v_positions[face_x1_z1.0[1] as usize];
                                        let x1_step = (x1_end - x1_start) / (h as f32);

                                        let z0_start = v_positions[face_x0_z0.0[3] as usize];
                                        let z0_end = v_positions[face_x1_z0.0[2] as usize];
                                        let z0_step = (z0_end - z0_start) / (w as f32);

                                        let z1_start = v_positions[face_x0_z1.0[0] as usize];
                                        let z1_end = v_positions[face_x1_z1.0[1] as usize];
                                        let z1_step = (z1_end - z1_start) / (w as f32);

                                        for x2 in 1..w {
                                            // z0 edge
                                            let index = calc_index!(x + x2, y, z) as usize + i;
                                            v_positions[index] = z0_start + z0_step * (x2 as f32);

                                            // z1 edge
                                            let index = calc_index!(x + x2, y, z + h) as usize + i;
                                            v_positions[index] = z1_start + z1_step * (x2 as f32);
                                        }

                                        for z2 in 1..h {
                                            // x0 edge
                                            let index = calc_index!(x, y, z + z2) as usize + i;
                                            v_positions[index] = x0_start + x0_step * (z2 as f32);

                                            // x1 edge
                                            let index = calc_index!(x + w, y, z + z2) as usize + i;
                                            v_positions[index] = x1_start + x1_step * (z2 as f32);
                                        }
                                    }
                                }
                            }

                            // YZ plane
                            {
                                let check_quad_usage = |y: usize, z: usize| -> bool {
                                    use_marks[(calc_index!(x, y, z) as usize + i) * 3 + 2]
                                };
                                let check_merge_marks = |y: usize, z: usize| -> bool {
                                    merge_marks[calc_index!(x, y, z) as usize + i]
                                        || merge_marks[calc_index!(x, y + 1, z) as usize + i]
                                        || merge_marks[calc_index!(x, y, z + 1) as usize + i]
                                        || merge_marks[calc_index!(x, y + 1, z + 1) as usize + i]
                                };

                                let index = (cell_index as usize + i) * 3 + 2;
                                let face = v_faces2[index];

                                if !check_quad_usage(y, z) {
                                    if let Some(face) = face {
                                        let normal = get_normal!(&face, index);

                                        let mut w = 1;
                                        let mut h = 1;

                                        if !check_merge_marks(y, z) {
                                            let mut prev_max_z2 = SECTOR_SIZE - z - 2;

                                            for y2 in 0..(SECTOR_SIZE - y - 2) {
                                                let mut max_z2 = 0;

                                                for z2 in 0..prev_max_z2 {
                                                    if check_quad_usage(y + y2, z + z2)
                                                        || check_merge_marks(y + y2, z + z2)
                                                        || !check_merge!(x, y + y2, z + z2, i, 2, normal)
                                                    {
                                                        break;
                                                    }

                                                    max_z2 += 1;
                                                }

                                                if (y2 * prev_max_z2) < ((y2 + 1) * max_z2) {
                                                    w = y2 + 1;
                                                    h = max_z2;
                                                    prev_max_z2 = max_z2;
                                                } else {
                                                    break;
                                                }
                                            }
                                        }

                                        // Mark used quads
                                        if w == 1 && h == 1 {
                                            let index = (calc_index!(x, y, z) as usize + i) * 3 + 2;
                                            use_marks[index] = true;
                                        } else {
                                            for y2 in 0..(w + 1) {
                                                for z2 in 0..(h + 1) {
                                                    let index = calc_index!(x, y + y2, z + z2) as usize + i;
                                                    merge_marks[index] = true;

                                                    if y2 < w && z2 < h {
                                                        use_marks[(index) * 3 + 2] = true;
                                                    }
                                                }
                                            }
                                        }

                                        // Create new quad
                                        let face_y0_z0 = &face;
                                        let face_y1_z0 = v_faces2
                                            [(calc_index!(x, y + w - 1, z) as usize + i) * 3 + 2]
                                            .as_ref()
                                            .unwrap();
                                        let face_y0_z1 = v_faces2
                                            [(calc_index!(x, y, z + h - 1) as usize + i) * 3 + 2]
                                            .as_ref()
                                            .unwrap();
                                        let face_y1_z1 = v_faces2
                                            [(calc_index!(x, y + w - 1, z + h - 1) as usize + i) * 3 + 2]
                                            .as_ref()
                                            .unwrap();

                                        if face.1 {
                                            v_faces.push([
                                                face_y0_z0.0[3],
                                                face_y1_z0.0[2],
                                                face_y1_z1.0[1],
                                                face_y0_z1.0[0],
                                            ]);
                                        } else {
                                            v_faces.push([
                                                face_y0_z1.0[0],
                                                face_y1_z1.0[1],
                                                face_y1_z0.0[2],
                                                face_y0_z0.0[3],
                                            ]);
                                        }

                                        /*v_faces.push([
                                            face_y0_z1[0],
                                            face_y1_z1[1],
                                            face_y1_z0[2],
                                            face_y0_z0[3],
                                        ]);*/

                                        // Set correct boundary vertices
                                        let y0_start = v_positions[face_y0_z0.0[3] as usize];
                                        let y0_end = v_positions[face_y0_z1.0[0] as usize];
                                        let y0_step = (y0_end - y0_start) / (h as f32);

                                        let y1_start = v_positions[face_y1_z0.0[2] as usize];
                                        let y1_end = v_positions[face_y1_z1.0[1] as usize];
                                        let y1_step = (y1_end - y1_start) / (h as f32);

                                        let z0_start = v_positions[face_y0_z0.0[3] as usize];
                                        let z0_end = v_positions[face_y1_z0.0[2] as usize];
                                        let z0_step = (z0_end - z0_start) / (w as f32);

                                        let z1_start = v_positions[face_y0_z1.0[0] as usize];
                                        let z1_end = v_positions[face_y1_z1.0[1] as usize];
                                        let z1_step = (z1_end - z1_start) / (w as f32);

                                        for y2 in 1..w {
                                            // z0 edge
                                            let index = calc_index!(x, y + y2, z) as usize + i;
                                            v_positions[index] = z0_start + z0_step * (y2 as f32);

                                            // z1 edge
                                            let index = calc_index!(x, y + y2, z + h) as usize + i;
                                            v_positions[index] = z1_start + z1_step * (y2 as f32);
                                        }

                                        for z2 in 1..h {
                                            // y0 edge
                                            let index = calc_index!(x, y, z + z2) as usize + i;
                                            v_positions[index] = y0_start + y0_step * (z2 as f32);

                                            // y1 edge
                                            let index = calc_index!(x, y + w, z + z2) as usize + i;
                                            v_positions[index] = y1_start + y1_step * (z2 as f32);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for i in 0..MAX_CELL_LAYERS {
                            for j in 0..3 {
                                let index = (cell_index as usize + i as usize) * 3 + j;
                                if let Some(face) = v_faces2[index] {
                                    if face.1 {
                                        v_faces.push([face.0[3], face.0[2], face.0[1], face.0[0]]);
                                    } else {
                                        v_faces.push([face.0[0], face.0[1], face.0[2], face.0[3]]);
                                    }
                                    //v_faces.push(face);
                                }
                            }
                        }
                    }
                }
            }
        }

        // TODO: separate triangulation and optimization into different functions

        // Generate mesh
        // -------------------------------------------------------------------------------------------------------------
        let mut index_map = [u32::MAX; ALIGNED_SECTOR_VOLUME * MAX_CELL_LAYERS];

        let mut vertices = Vec::with_capacity(SECTOR_VOLUME * MAX_CELL_LAYERS);
        let mut indices = Vec::with_capacity(v_faces.len() * 6);

        for face in &v_faces {
            let face_indices = [
                face[0] as usize,
                face[1] as usize,
                face[2] as usize,
                face[0] as usize,
                face[2] as usize,
                face[3] as usize,
            ];

            // Validate face
            let mut invalid_face = false;
            for &face_index in &face_indices {
                if v_positions[face_index][0].is_nan() {
                    invalid_face = true;
                    break;
                }
            }
            if invalid_face {
                continue;
            }

            // Add vertices & indices
            for &face_index in &face_indices {
                if index_map[face_index] == u32::MAX {
                    index_map[face_index] = vertices.len() as u32;

                    vertices.push(Vertex {
                        position: v_positions[face_index],
                        density: v_density_indices[face_index],
                    });
                }

                indices.push(index_map[face_index]);
            }
        }

        (vertices, indices)
    }

    fn triangulate2(&mut self, sector_pos: [u8; 3], details: f32) -> (Vec<Vertex>, Vec<u32>) {
        let sector = &self.sectors[sector_pos[0] as usize][sector_pos[1] as usize][sector_pos[2] as usize];
        let densities = &sector.densities;
        let density_indices = &sector.indices;

        let mut field = vec![0.0; (SECTOR_SIZE + 1) * (SECTOR_SIZE + 1) * (SECTOR_SIZE + 1)];

        macro_rules! field_index {
            ($x: expr, $y: expr, $z: expr) => {
                ($x * (SECTOR_SIZE + 1) * (SECTOR_SIZE + 1) + $y * (SECTOR_SIZE + 1) + $z) as usize
            };
        }

        for x in 0..(SECTOR_SIZE + 1) {
            for y in 0..(SECTOR_SIZE + 1) {
                for z in 0..(SECTOR_SIZE + 1) {
                    let density_index = density_indices[x][y][z];

                    let index = density_index & 0x00ffffff;
                    let count = density_index >> 24;

                    let density = if count > 0 {
                        densities[index as usize].density
                    } else {
                        0
                    };

                    field[field_index!(x, y, z)] = (density as f32) / 255.0;
                }
            }
        }

        let (vertices, oct) = dc::contour::construct_octree(&field, SECTOR_SIZE as u32, 0.5);
        let indices = dc::contour::generate_mesh(&oct);

        let vertices: Vec<Vertex> = vertices
            .iter()
            .map(|pos| Vertex {
                position: *pos,
                density: 0,
            })
            .collect();

        (vertices, indices)
    }

    pub fn update_mesh(&mut self, lod: f32) {
        let mut changed = false;

        let mut prev_vertex_count = 0;
        let mut prev_index_count = 0;

        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    let sector = &self.sectors[x][y][z];
                    prev_vertex_count += sector.vertex_count;
                    prev_index_count += sector.index_count;

                    if sector.changed {
                        changed = true;
                        break;
                    }
                }
            }
        }

        if !changed {
            return;
        }

        let mut vertices = Vec::with_capacity(prev_vertex_count as usize);
        let mut indices = Vec::with_capacity(prev_index_count as usize);

        // Collect vertices & indices
        for x in 0..SIZE_IN_SECTORS {
            for y in 0..SIZE_IN_SECTORS {
                for z in 0..SIZE_IN_SECTORS {
                    let sector_changed = self.sectors[x][y][z].changed;

                    let (mut sector_vertices, sector_indices) = if sector_changed {
                        let (sector_vertices, mut sector_indices) =
                            self.triangulate2([x as u8, y as u8, z as u8], lod);

                        // Adjust indices
                        for index in &mut sector_indices {
                            *index += vertices.len() as u32;
                        }

                        (sector_vertices, sector_indices)
                    } else {
                        let sector = &self.sectors[x][y][z];
                        let sector_vertices = self
                            .vertex_mesh
                            .get_vertices(sector.vertices_offset, sector.vertex_count);
                        let mut sector_indices = self
                            .vertex_mesh
                            .get_indices(sector.indices_offset, sector.index_count);

                        // Adjust indices
                        for index in &mut sector_indices {
                            *index -= sector.vertices_offset;
                            *index += vertices.len() as u32;
                        }

                        (sector_vertices, sector_indices)
                    };

                    // Adjust vertices
                    for v in &mut sector_vertices {
                        v.position += na::Vector3::new(x as f32, y as f32, z as f32) * (SECTOR_SIZE as f32);
                    }

                    vertices.extend(sector_vertices);
                    indices.extend(sector_indices);

                    if sector_changed {
                        self.sectors[x][y][z].changed = false;
                    }
                }
            }
        }

        self.vertex_mesh.set_vertices(&vertices, Some(&indices));
    }
}

pub fn new(device: &Arc<vkw::Device>) -> Cluster {
    Cluster {
        sectors: Default::default(),
        vertex_mesh: device.create_vertex_mesh().unwrap(),
    }
}

impl specs::Component for Cluster {
    type Storage = specs::VecStorage<Self>;
}
