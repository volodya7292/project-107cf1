use crate::octree;
use crate::octree::Octree;
use crate::utils;
use nalgebra as na;

#[derive(Copy, Clone)]
pub struct NodeData<T> {
    pub corners: u8,
    pub vertex_pos: na::Vector3<f32>,
    pub vertex_index: u32,
    pub is_seam: bool,
    pub data: [T; 8],
}

#[derive(Copy, Clone)]
pub struct NodeDataDiscrete<T> {
    pub corners: u8,
    pub densities: [f32; 8],
    pub vertex_pos: Option<na::Vector3<f32>>,
    pub data: [T; 8],
}

impl<T> NodeDataDiscrete<T> {
    pub fn new(densities: [f32; 8], iso_value: f32, data: [T; 8]) -> NodeDataDiscrete<T> {
        let mut corners = 0_u8;

        for i in 0..8 {
            corners |= ((densities[i] > iso_value) as u8) << i;
        }

        if corners == 0 || corners == 0xff {
            return NodeDataDiscrete {
                corners,
                densities,
                vertex_pos: None,
                data,
            };
        }

        let mut avg_pos = na::Vector3::<f32>::new(0.0, 0.0, 0.0);
        let mut edge_count = 0;

        for i in 0..12 {
            let di0 = EDGE_VERT_MAP[i][0];
            let di1 = EDGE_VERT_MAP[i][1];
            let d0 = densities[di0];
            let d1 = densities[di1];

            if (d0 > iso_value) == (d1 > iso_value) {
                continue;
            }

            let offset0 = CELL_OFFSETS[di0];
            let offset1 = CELL_OFFSETS[di1];

            let pos0 = na::Vector3::new(offset0[0] as f32, offset0[1] as f32, offset0[2] as f32);
            let pos1 = na::Vector3::new(offset1[0] as f32, offset1[1] as f32, offset1[2] as f32);

            let interpolation = (iso_value - d0) / (d1 - d0);
            let pos = pos0 + (pos1 - pos0) * interpolation;

            avg_pos += pos;
            edge_count += 1;

            if edge_count >= 6 {
                break;
            }
        }

        avg_pos /= edge_count as f32;

        NodeDataDiscrete {
            corners,
            densities,
            vertex_pos: Some(avg_pos),
            data,
        }
    }
}

const CELL_OFFSETS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
];

const EDGE_VERT_MAP: [[usize; 2]; 12] = [
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
];

const PROCESS_EDGE_MASK: [[usize; 4]; 3] = [[3, 2, 1, 0], [7, 5, 6, 4], [11, 10, 9, 8]];

const EDGE_PROC_EDGE_MASK: [[[usize; 5]; 2]; 3] = [
    [[3, 2, 1, 0, 0], [7, 6, 5, 4, 0]],
    [[5, 1, 4, 0, 1], [7, 3, 6, 2, 1]],
    [[6, 4, 2, 0, 2], [7, 5, 3, 1, 2]],
];

const FACE_PROC_FACE_MASK: [[[usize; 3]; 4]; 3] = [
    [[4, 0, 0], [5, 1, 0], [6, 2, 0], [7, 3, 0]],
    [[2, 0, 1], [6, 4, 1], [3, 1, 1], [7, 5, 1]],
    [[1, 0, 2], [3, 2, 2], [5, 4, 2], [7, 6, 2]],
];
const FACE_PROC_EDGE_MASK: [[[usize; 6]; 4]; 3] = [
    [
        [1, 4, 0, 5, 1, 1],
        [1, 6, 2, 7, 3, 1],
        [0, 4, 6, 0, 2, 2],
        [0, 5, 7, 1, 3, 2],
    ],
    [
        [0, 2, 3, 0, 1, 0],
        [0, 6, 7, 4, 5, 0],
        [1, 2, 0, 6, 4, 2],
        [1, 3, 1, 7, 5, 2],
    ],
    [
        [1, 1, 0, 3, 2, 0],
        [1, 5, 4, 7, 6, 0],
        [0, 1, 5, 0, 4, 1],
        [0, 3, 7, 2, 6, 1],
    ],
];

const CELL_PROC_FACE_MASK: [[usize; 3]; 12] = [
    [0, 4, 0],
    [1, 5, 0],
    [2, 6, 0],
    [3, 7, 0],
    [0, 2, 1],
    [4, 6, 1],
    [1, 3, 1],
    [5, 7, 1],
    [0, 1, 2],
    [2, 3, 2],
    [4, 5, 2],
    [6, 7, 2],
];
const CELL_PROC_EDGE_MASK: [[usize; 5]; 6] = [
    [0, 1, 2, 3, 0],
    [4, 5, 6, 7, 0],
    [0, 4, 1, 5, 1],
    [2, 6, 3, 7, 1],
    [0, 2, 4, 6, 2],
    [1, 3, 5, 7, 2],
];

/// Construct octree from field of size (dim_size + 1) x (dim_size + 1) x (dim_size + 1)
///
/// dim_size: power of 2
pub fn construct_octree<T>(
    field: &[(f32, T)],
    dim_size: u32,
    iso_value: f32,
) -> (Vec<na::Vector3<f32>>, octree::Octree<NodeData<T>>)
where
    T: Clone,
{
    let a_dim_size = dim_size + 1;

    assert_eq!(field.len() as u32, a_dim_size * a_dim_size * a_dim_size);
    assert!(utils::is_pow_of_2(dim_size as u64));

    let depth = utils::log2(dim_size) + 1;
    let mut oct = octree::with_capacity::<NodeData<T>>(dim_size, (8_u32.pow(depth) - 1) / 7);
    let mut vertices = vec![];
    macro_rules! field_index {
        ($x: expr, $y: expr, $z: expr) => {
            ($x * a_dim_size * a_dim_size + $y * a_dim_size + $z) as usize
        };
    }

    for x in 0..dim_size {
        for y in 0..dim_size {
            for z in 0..dim_size {
                let points = [
                    field[field_index!(x, y, z)].clone(),
                    field[field_index!(x, y, z + 1)].clone(),
                    field[field_index!(x, y + 1, z)].clone(),
                    field[field_index!(x, y + 1, z + 1)].clone(),
                    field[field_index!(x + 1, y, z)].clone(),
                    field[field_index!(x + 1, y, z + 1)].clone(),
                    field[field_index!(x + 1, y + 1, z)].clone(),
                    field[field_index!(x + 1, y + 1, z + 1)].clone(),
                ];
                let data = [
                    points[0].1.clone(),
                    points[1].1.clone(),
                    points[2].1.clone(),
                    points[3].1.clone(),
                    points[4].1.clone(),
                    points[5].1.clone(),
                    points[6].1.clone(),
                    points[7].1.clone(),
                ];

                let mut corners = 0_u8;
                for i in 0..8 {
                    corners |= ((points[i].0 > iso_value) as u8) << i;
                }
                if corners == 0 || corners == 0xff {
                    continue;
                }

                let mut avg_pos = na::Vector3::<f32>::new(0.0, 0.0, 0.0);
                let mut edge_count = 0;

                for i in 0..12 {
                    let di0 = EDGE_VERT_MAP[i][0];
                    let di1 = EDGE_VERT_MAP[i][1];
                    let d0 = points[di0].0;
                    let d1 = points[di1].0;

                    if (d0 > iso_value) == (d1 > iso_value) {
                        continue;
                    }

                    let offset0 = CELL_OFFSETS[di0];
                    let offset1 = CELL_OFFSETS[di1];

                    let pos0 = na::Vector3::new(offset0[0] as f32, offset0[1] as f32, offset0[2] as f32);
                    let pos1 = na::Vector3::new(offset1[0] as f32, offset1[1] as f32, offset1[2] as f32);

                    let interpolation = (iso_value - d0) / (d1 - d0);
                    let pos = pos0 + (pos1 - pos0) * interpolation;

                    avg_pos += pos;
                    edge_count += 1;

                    if edge_count >= 6 {
                        break;
                    }
                }

                avg_pos /= edge_count as f32;

                let is_seam_node = (x == dim_size - 1) || (y == dim_size - 1) || (z == dim_size - 1);

                oct.set_node(
                    na::Vector3::new(x, y, z),
                    octree::Node::new_leaf(
                        1,
                        NodeData {
                            corners,
                            vertex_pos: avg_pos,
                            vertex_index: vertices.len() as u32,
                            is_seam: is_seam_node,
                            data,
                        },
                    ),
                );
                vertices.push(avg_pos);
            }
        }
    }

    (vertices, oct)
}

pub fn construct_nodes<T>(
    field: &[Option<(f32, T)>],
    dim_size: u32,
    node_size: u32,
    iso_value: f32,
) -> Vec<octree::LeafNode<NodeData<T>>>
where
    T: Default + Copy,
{
    let a_dim_size = dim_size + 1;
    assert_eq!(field.len() as u32, a_dim_size * a_dim_size * a_dim_size);

    macro_rules! field_index {
        ($x: expr, $y: expr, $z: expr) => {
            ($x * a_dim_size * a_dim_size + $y * a_dim_size + $z) as usize
        };
    }

    let mut nodes = Vec::with_capacity((dim_size * dim_size * dim_size) as usize);
    let mut temp_densities = [0.0_f32; 8];
    let mut temp_data = [T::default(); 8];

    for x in 0..dim_size {
        for y in 0..dim_size {
            for z in 0..dim_size {
                let mut valid_cell = true;

                for i in 0..8_u32 {
                    let (x2, y2, z2) = ((i / 4) % 2, (i / 2) % 2, i % 2);

                    if let Some(point) = field[field_index!(x + x2, y + y2, z + z2)] {
                        temp_densities[i as usize] = point.0;
                        temp_data[i as usize] = point.1;
                    } else {
                        valid_cell = false;
                        break;
                    }
                }

                if !valid_cell {
                    continue;
                }

                let mut corners = 0_u8;
                for i in 0..8 {
                    corners |= ((temp_densities[i] > iso_value) as u8) << i;
                }
                if corners == 0 || corners == 0xff {
                    continue;
                }

                let mut avg_pos = na::Vector3::<f32>::new(0.0, 0.0, 0.0);
                let mut edge_count = 0;

                for i in 0..12 {
                    let di0 = EDGE_VERT_MAP[i][0];
                    let di1 = EDGE_VERT_MAP[i][1];
                    let d0 = temp_densities[di0];
                    let d1 = temp_densities[di1];

                    if (d0 > iso_value) == (d1 > iso_value) {
                        continue;
                    }

                    let offset0 = CELL_OFFSETS[di0];
                    let offset1 = CELL_OFFSETS[di1];

                    let pos0 = na::Vector3::new(offset0[0] as f32, offset0[1] as f32, offset0[2] as f32);
                    let pos1 = na::Vector3::new(offset1[0] as f32, offset1[1] as f32, offset1[2] as f32);

                    let interpolation = (iso_value - d0) / (d1 - d0);
                    let pos = pos0 + (pos1 - pos0) * interpolation;

                    avg_pos += pos;
                    edge_count += 1;

                    if edge_count >= 6 {
                        break;
                    }
                }

                avg_pos /= edge_count as f32;

                let is_seam_node = (x == dim_size - 1) || (y == dim_size - 1) || (z == dim_size - 1);

                nodes.push(octree::LeafNode::new(
                    na::Vector3::new(x * node_size, y * node_size, z * node_size),
                    node_size,
                    NodeData {
                        corners,
                        vertex_pos: avg_pos,
                        vertex_index: u32::MAX,
                        is_seam: is_seam_node,
                        data: temp_data,
                    },
                ));
            }
        }
    }

    nodes
}

fn process_edge<T>(nodes: &[octree::Node<NodeData<T>>; 4], dir: usize, indices_out: &mut Vec<u32>) {
    let mut min_size = u32::MAX;
    let mut min_index = 0;
    let mut flip = false;
    let mut indices = [0u32; 4];
    let mut sign_change = [false; 4];

    let mut all_seam = true;

    for i in 0..4 {
        let node = &nodes[i];

        let node_data = if let octree::NodeType::Leaf(data) = node.ty() {
            data
        } else {
            unreachable!()
        };

        if !node_data.is_seam {
            all_seam = false;
        }

        let edge = PROCESS_EDGE_MASK[dir][i];
        let di0 = EDGE_VERT_MAP[edge][0];
        let di1 = EDGE_VERT_MAP[edge][1];

        let m0 = ((node_data.corners >> di0) & 1) == 1;
        let m1 = ((node_data.corners >> di1) & 1) == 1;

        if node.size() < min_size {
            min_size = node.size();
            min_index = i;
            flip = m1;
        }

        indices[i] = node_data.vertex_index;
        sign_change[i] = m0 ^ m1;
    }

    // Avoid processing an edge that completely consists of seam nodes.
    // Seam nodes are only used to extend non-seam nodes.
    if all_seam {
        return;
    }

    if sign_change[min_index] {
        if flip {
            indices_out.extend_from_slice(&[
                indices[0], indices[3], indices[1], indices[0], indices[2], indices[3],
            ]);
        } else {
            indices_out.extend_from_slice(&[
                indices[0], indices[1], indices[3], indices[0], indices[3], indices[2],
            ]);
        }
    }
}

fn edge_proc<T>(
    oct: &Octree<NodeData<T>>,
    nodes: &[octree::Node<NodeData<T>>; 4],
    dir: usize,
    indices_out: &mut Vec<u32>,
) where
    T: Clone,
{
    let mut all_leaf = true;

    for i in 0..4 {
        if let octree::NodeType::Internal(_) = nodes[i].ty() {
            all_leaf = false;
            break;
        }
    }

    if all_leaf {
        process_edge(&nodes, dir, indices_out);
    } else {
        for i in 0..2 {
            let mut edge_nodes = nodes.clone();
            let mut success = true;

            for j in 0..4 {
                if let octree::NodeType::Internal(children) = edge_nodes[j].ty() {
                    let child_id = children[EDGE_PROC_EDGE_MASK[dir][i][j]];

                    if child_id == octree::NODE_ID_NONE {
                        success = false;
                        break;
                    }

                    edge_nodes[j] = oct.get_node(child_id).clone().unwrap();
                }
            }

            if success {
                edge_proc(oct, &edge_nodes, EDGE_PROC_EDGE_MASK[dir][i][4], indices_out);
            }
        }
    }
}

fn face_proc<T>(
    oct: &Octree<NodeData<T>>,
    nodes: &[octree::Node<NodeData<T>>; 2],
    dir: usize,
    indices_out: &mut Vec<u32>,
) where
    T: Clone,
{
    let mut internal_present = false;

    for i in 0..2 {
        if let octree::NodeType::Internal(_) = nodes[i].ty() {
            internal_present = true;
            break;
        }
    }
    if !internal_present {
        return;
    }

    for i in 0..4 {
        let mut face_nodes = nodes.clone();
        let mut success = true;

        for j in 0..2 {
            if let octree::NodeType::Internal(children) = face_nodes[j].ty() {
                let child_id = children[FACE_PROC_FACE_MASK[dir][i][j]];

                if child_id == octree::NODE_ID_NONE {
                    success = false;
                    break;
                }

                face_nodes[j] = oct.get_node(child_id).clone().unwrap();
            }
        }

        if success {
            face_proc(oct, &face_nodes, FACE_PROC_FACE_MASK[dir][i][2], indices_out);
        }
    }

    const ORDERS: [[usize; 4]; 2] = [[0, 0, 1, 1], [0, 1, 0, 1]];

    for i in 0..4 {
        let order = &ORDERS[FACE_PROC_EDGE_MASK[dir][i][0]];
        let mut edge_nodes = [
            nodes[order[0]].clone(),
            nodes[order[1]].clone(),
            nodes[order[2]].clone(),
            nodes[order[3]].clone(),
        ];
        let mut success = true;

        for j in 0..4 {
            if let octree::NodeType::Internal(children) = edge_nodes[j].ty() {
                let child_id = children[FACE_PROC_EDGE_MASK[dir][i][1 + j]];

                if child_id == octree::NODE_ID_NONE {
                    success = false;
                    break;
                }

                edge_nodes[j] = oct.get_node(child_id).clone().unwrap();
            }
        }

        if success {
            edge_proc(oct, &edge_nodes, FACE_PROC_EDGE_MASK[dir][i][5], indices_out);
        }
    }
}

fn cell_proc<T>(oct: &Octree<NodeData<T>>, node: &octree::Node<NodeData<T>>, indices_out: &mut Vec<u32>)
where
    T: Clone,
{
    if let octree::NodeType::Internal(children) = &node.ty() {
        for i in 0..8 {
            let child_id = children[i];

            if child_id != octree::NODE_ID_NONE {
                cell_proc(oct, &oct.get_node(child_id).clone().unwrap(), indices_out);
            }
        }

        for i in 0..12 {
            let ids = [
                children[CELL_PROC_FACE_MASK[i][0]],
                children[CELL_PROC_FACE_MASK[i][1]],
            ];
            if ids[0] == octree::NODE_ID_NONE || ids[1] == octree::NODE_ID_NONE {
                continue;
            }

            let face_nodes = [
                oct.get_node(ids[0]).clone().unwrap(),
                oct.get_node(ids[1]).clone().unwrap(),
            ];

            face_proc(oct, &face_nodes, CELL_PROC_FACE_MASK[i][2], indices_out);
        }

        for i in 0..6 {
            let ids = [
                children[CELL_PROC_EDGE_MASK[i][0]],
                children[CELL_PROC_EDGE_MASK[i][1]],
                children[CELL_PROC_EDGE_MASK[i][2]],
                children[CELL_PROC_EDGE_MASK[i][3]],
            ];
            if ids[0] == octree::NODE_ID_NONE
                || ids[1] == octree::NODE_ID_NONE
                || ids[2] == octree::NODE_ID_NONE
                || ids[3] == octree::NODE_ID_NONE
            {
                continue;
            }

            let edge_nodes = [
                oct.get_node(ids[0]).clone().unwrap(),
                oct.get_node(ids[1]).clone().unwrap(),
                oct.get_node(ids[2]).clone().unwrap(),
                oct.get_node(ids[3]).clone().unwrap(),
            ];

            edge_proc(oct, &edge_nodes, CELL_PROC_EDGE_MASK[i][4], indices_out);
        }
    }
}

pub fn generate_mesh<T>(oct: &Octree<NodeData<T>>) -> Vec<u32>
where
    T: Clone,
{
    let oct_size = oct.size() as usize;
    let mut indices = Vec::<u32>::with_capacity(oct_size * oct_size * oct_size * 18);

    if let Some(node) = oct.get_node(0) {
        cell_proc(oct, node, &mut indices);
    }

    indices
}
