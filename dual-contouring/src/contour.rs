use crate::octree;
use crate::octree::Octree;
use crate::utils;
use nalgebra as na;

#[derive(Copy, Clone)]
pub struct NodeData {
    pub corners: u8,
    pub vertex_pos: na::Vector3<f32>,
    pub vertex_index: u32,
}

#[derive(Copy, Clone)]
pub struct NodeDataDiscrete {
    pub corners: u8,
    pub densities: [f32; 8],
    pub vertex_pos: Option<na::Vector3<f32>>,
}

impl NodeDataDiscrete {
    pub fn new(node_pos: &na::Vector3<u32>, densities: [f32; 8], iso_value: f32) -> NodeDataDiscrete {
        let mut corners = 0_u8;

        for i in 0..8 {
            corners |= ((densities[i] > iso_value) as u8) << i;
        }

        if corners == 0 || corners == 0xff {
            return NodeDataDiscrete {
                corners,
                densities,
                vertex_pos: None,
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

            let pos0 = na::Vector3::new(
                (node_pos.x + offset0[0]) as f32,
                (node_pos.y + offset0[1]) as f32,
                (node_pos.z + offset0[2]) as f32,
            );
            let pos1 = na::Vector3::new(
                (node_pos.x + offset1[0]) as f32,
                (node_pos.y + offset1[1]) as f32,
                (node_pos.z + offset1[2]) as f32,
            );

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
pub fn construct_octree(
    field: &[f32],
    dim_size: u32,
    iso_value: f32,
) -> (Vec<na::Vector3<f32>>, octree::Octree<NodeData>) {
    let a_dim_size = dim_size + 1;

    assert_eq!(field.len() as u32, a_dim_size * a_dim_size * a_dim_size);
    assert!(utils::is_pow_of_2(dim_size as u64));

    let depth = utils::log2(dim_size) + 1;
    let mut oct = octree::with_capacity::<NodeData>(dim_size, (8_u32.pow(depth) - 1) / 7);
    let mut vertices = vec![];
    macro_rules! field_index {
        ($x: expr, $y: expr, $z: expr) => {
            ($x * a_dim_size * a_dim_size + $y * a_dim_size + $z) as usize
        };
    }

    for x in 0..dim_size {
        for y in 0..dim_size {
            for z in 0..dim_size {
                let densities = [
                    field[field_index!(x, y, z)],
                    field[field_index!(x, y, z + 1)],
                    field[field_index!(x, y + 1, z)],
                    field[field_index!(x, y + 1, z + 1)],
                    field[field_index!(x + 1, y, z)],
                    field[field_index!(x + 1, y, z + 1)],
                    field[field_index!(x + 1, y + 1, z)],
                    field[field_index!(x + 1, y + 1, z + 1)],
                ];

                let mut corners = 0_u8;
                for i in 0..8 {
                    corners |= ((densities[i] > iso_value) as u8) << i;
                }
                if corners == 0 || corners == 0xff {
                    continue;
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

                    let pos0 = na::Vector3::new(
                        (x + offset0[0]) as f32,
                        (y + offset0[1]) as f32,
                        (z + offset0[2]) as f32,
                    );
                    let pos1 = na::Vector3::new(
                        (x + offset1[0]) as f32,
                        (y + offset1[1]) as f32,
                        (z + offset1[2]) as f32,
                    );

                    let interpolation = (iso_value - d0) / (d1 - d0);
                    let pos = pos0 + (pos1 - pos0) * interpolation;

                    avg_pos += pos;
                    edge_count += 1;

                    if edge_count >= 6 {
                        break;
                    }
                }

                avg_pos /= edge_count as f32;

                oct.set_node(
                    na::Vector3::new(x, y, z),
                    octree::Node::new_leaf(
                        1,
                        NodeData {
                            corners,
                            vertex_pos: avg_pos,
                            vertex_index: vertices.len() as u32,
                        },
                    ),
                );
                vertices.push(avg_pos);
            }
        }
    }

    (vertices, oct)
}

pub fn construct_nodes(
    field: &[Option<f32>],
    dim_size: u32,
    node_size: u32,
    iso_value: f32,
) -> Vec<octree::LeafNode<NodeData>> {
    let a_dim_size = dim_size + 1;
    assert_eq!(field.len() as u32, a_dim_size * a_dim_size * a_dim_size);

    macro_rules! field_index {
        ($x: expr, $y: expr, $z: expr) => {
            ($x * a_dim_size * a_dim_size + $y * a_dim_size + $z) as usize
        };
    }

    let mut nodes = Vec::with_capacity((dim_size * dim_size * dim_size) as usize);
    let mut temp_densities = [0.0_f32; 8];

    for x in 0..dim_size {
        for y in 0..dim_size {
            for z in 0..dim_size {
                let mut valid_cell = true;

                for i in 0..8_u32 {
                    let (x2, y2, z2) = ((i / 4) % 2, (i / 2) % 2, i % 2);

                    if let Some(density) = field[field_index!(x + x2, y + y2, z + z2)] {
                        temp_densities[i as usize] = density;
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

                    let pos0 = na::Vector3::new(
                        (x + offset0[0]) as f32,
                        (y + offset0[1]) as f32,
                        (z + offset0[2]) as f32,
                    );
                    let pos1 = na::Vector3::new(
                        (x + offset1[0]) as f32,
                        (y + offset1[1]) as f32,
                        (z + offset1[2]) as f32,
                    );

                    let interpolation = (iso_value - d0) / (d1 - d0);
                    let pos = pos0 + (pos1 - pos0) * interpolation;

                    avg_pos += pos;
                    edge_count += 1;

                    if edge_count >= 6 {
                        break;
                    }
                }

                avg_pos /= edge_count as f32;

                nodes.push(octree::LeafNode::new(
                    na::Vector3::new(x * node_size, y * node_size, z * node_size),
                    node_size,
                    NodeData {
                        corners,
                        vertex_pos: avg_pos * (node_size as f32),
                        vertex_index: u32::MAX,
                    },
                ));
            }
        }
    }

    nodes
}

fn process_edge(nodes: &[octree::Node<NodeData>; 4], dir: usize, indices_out: &mut Vec<u32>) {
    let mut min_size = u32::MAX;
    let mut min_index = 0;
    let mut flip = false;
    let mut indices = [0u32; 4];
    let mut sign_change = [false; 4];

    for i in 0..4 {
        let node = &nodes[i];

        let node_data = if let octree::NodeType::Leaf(data) = node.ty() {
            data
        } else {
            unreachable!()
        };

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

fn edge_proc(
    oct: &Octree<NodeData>,
    nodes: &[octree::Node<NodeData>; 4],
    dir: usize,
    indices_out: &mut Vec<u32>,
) {
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

fn face_proc(
    oct: &Octree<NodeData>,
    nodes: &[octree::Node<NodeData>; 2],
    dir: usize,
    indices_out: &mut Vec<u32>,
) {
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

fn cell_proc(oct: &Octree<NodeData>, node: &octree::Node<NodeData>, indices_out: &mut Vec<u32>) {
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

pub fn generate_mesh(oct: &Octree<NodeData>) -> Vec<u32> {
    let mut indices = Vec::<u32>::new();

    if let Some(node) = oct.root_node() {
        cell_proc(oct, node, &mut indices);
    }

    indices
}
