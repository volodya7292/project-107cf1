use nalgebra as na;
use std::{mem::MaybeUninit, ptr};

const COLLAPSE_MAX_DEGREE: u32 = 16;

#[derive(Copy, Clone)]
struct Vertex {
    pos: na::Vector3<f32>,
}

impl Vertex {
    const POS_SIZE: usize = 12;
}
pub struct VertexReader<'a> {
    data: &'a [u8],
    stride: usize,
    position_offset: usize,
}

impl VertexReader<'_> {
    pub fn new<R>(data: &[u8], stride: usize, position_offset: usize) -> VertexReader {
        if position_offset + Vertex::POS_SIZE > stride {
            panic!(
                "position_offset ({}) + Vertex::POS_SIZE > stride ({})",
                position_offset, stride
            );
        }

        VertexReader {
            data,
            stride,
            position_offset,
        }
    }

    fn vertex_count(&self) -> usize {
        self.data.len() / self.stride
    }

    fn read_vertices(&self) -> Vec<Vertex> {
        (0..self.vertex_count())
            .map(|i| {
                let ptr = unsafe { self.data.as_ptr().offset((self.stride * i) as isize) };
                let mut pos = MaybeUninit::<na::Vector3<f32>>::uninit();

                unsafe {
                    ptr::copy_nonoverlapping(
                        ptr.offset(self.position_offset as isize),
                        pos.as_mut_ptr() as *mut u8,
                        12,
                    );
                }

                Vertex {
                    pos: unsafe { pos.assume_init() },
                }
            })
            .collect::<Vec<Vertex>>()
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
union Edge {
    id: u64,
    v_ids: (u32, u32),
}

pub struct Options {
    // Each iteration involves selecting a fraction of the edges at random as possible
    // candidates for collapsing. There is likely a sweet spot here trading off against number
    // of edges processed vs number of invalid collapses generated due to collisions
    // (the more edges that are processed the higher the chance of collisions happening)
    edge_fraction: f32,
    /// Stop simplfying after a given number of iterations
    max_iterations: usize,
    /// And/or stop simplifying when we've reached a threshold of the input triangles
    target_triangle_count: usize,
    // Useful for controlling how uniform the mesh is (or isn't)
    max_edge_size: f32,
}

fn build_candidate_edges(vertex_count: usize, indices: &[u32]) -> Vec<Edge> {
    let mut edges = Vec::<Edge>::with_capacity(indices.len());

    for i in (0..indices.len()).step_by(3) {
        let index0 = indices[i];
        let index1 = indices[i + 1];
        let index2 = indices[i + 2];

        edges.push(Edge {
            v_ids: (index0.min(index1), index0.max(index1)),
        });
        edges.push(Edge {
            v_ids: (index1.min(index2), index1.max(index2)),
        });
        edges.push(Edge {
            v_ids: (index0.min(index2), index0.max(index2)),
        });
    }

    edges.sort_by(|a, b| unsafe { a.id.cmp(&b.id) });

    let mut filtered_edges = Vec::<Edge>::with_capacity(edges.len());
    let mut boundary_verts = vec![false; vertex_count];
    let mut prev_edge = edges[0];
    let mut count = 1;

    for i in 1..edges.len() {
        let curr_edge = edges[i];

        if unsafe { curr_edge.id != prev_edge.id } {
            if count == 1 {
                boundary_verts[unsafe { prev_edge.v_ids.0 } as usize] = true;
                boundary_verts[unsafe { prev_edge.v_ids.1 } as usize] = true;
            } else {
                filtered_edges.push(prev_edge);
            }
            count = 1;
        } else {
            count += 1;
        }

        prev_edge = curr_edge;
    }

    edges.clear();

    for edge in &filtered_edges {
        if !boundary_verts[unsafe { edge.v_ids.0 } as usize]
            && !boundary_verts[unsafe { edge.v_ids.1 } as usize]
        {
            edges.push(*edge);
        }
    }

    edges
}

fn find_valid_collapses(
    vertices: &[Vertex],
    vertex_triangle_counts: &[u32],
    edges: &[Edge],
    options: &Options,
    collapse_positions: &mut [na::Vector3<f32>],
    collapse_edge_ids: &mut [u32],
) -> Vec<u32> {
    let step = (1.0 / options.edge_fraction).round() as usize;
    let target_valid_edges = (edges.len() as f64 * options.edge_fraction as f64) as usize;

    let mut valid_collapses = Vec::<u32>::with_capacity(edges.len());

    for i in 0..step {
        let mut state = false;

        for j in (i..edges.len()).step_by(step) {
            if valid_collapses.len() >= target_valid_edges {
                state = true;
                break;
            }

            let edge = &edges[j];
            let v0 = vertices[unsafe { edge.v_ids.0 } as usize];
            let v1 = vertices[unsafe { edge.v_ids.1 } as usize];

            if (v1.pos - v0.pos).magnitude() > options.max_edge_size {
                continue;
            }

            let degree = vertex_triangle_counts[unsafe { edge.v_ids.0 } as usize]
                + vertex_triangle_counts[unsafe { edge.v_ids.1 } as usize];
            if degree > COLLAPSE_MAX_DEGREE {
                continue;
            }

            valid_collapses.push(j as u32);
            collapse_positions[j] = (v0.pos + v1.pos) * 0.5;
            collapse_edge_ids[unsafe { edge.v_ids.0 } as usize] = j as u32;
            collapse_edge_ids[unsafe { edge.v_ids.1 } as usize] = j as u32;
        }

        if state {
            break;
        }
    }

    valid_collapses
}

fn collapse_edges(
    valid_collapses: &[u32],
    edges: &[Edge],
    collapse_edge_ids: &mut [u32],
    collapse_target: &mut [u32],
    collapse_positions: &mut [na::Vector3<f32>],
    vertices: &mut [Vertex],
) {
    for &i in valid_collapses {
        let edge = &edges[i as usize];
        let min = unsafe { edge.v_ids.0 } as usize;
        let max = unsafe { edge.v_ids.1 } as usize;

        if (collapse_edge_ids[min] == i) && (collapse_edge_ids[max] == i) {
            collapse_target[max] = min as u32;
            vertices[min].pos = collapse_positions[i as usize];
        }
    }
}

fn remove_triangles(
    vertices: &[Vertex],
    collapse_target: &mut [u32],
    indices: &mut Vec<u32>,
    vertex_triangle_counts: &mut Vec<u32>,
) {
    let mut temp_indices = Vec::<u32>::with_capacity(indices.len());
    *vertex_triangle_counts = vec![0_u32; vertices.len()];

    for i in (0..indices.len()).step_by(3) {
        let ind = &mut indices[i..(i + 3)];

        for j in 0..3 {
            let index = &mut ind[j];
            let t = collapse_target[*index as usize];

            if t != u32::MAX {
                *index = t;
            }
        }

        if (ind[0] == ind[1]) || (ind[0] == ind[2]) || (ind[1] == ind[2]) {
            continue;
        }

        for j in 0..3 {
            vertex_triangle_counts[ind[j] as usize] += 1;
        }

        temp_indices.extend(ind.iter());
    }

    *indices = temp_indices;
}

fn remove_edges(collapse_target: &[u32], edges: &mut Vec<Edge>) {
    let mut temp_edges = Vec::<Edge>::with_capacity(edges.len());

    for edge in edges.iter_mut() {
        let t = collapse_target[unsafe { edge.v_ids.0 } as usize];
        if t != u32::MAX {
            edge.v_ids.0 = t;
        }

        let t = collapse_target[unsafe { edge.v_ids.1 } as usize];
        if t != u32::MAX {
            edge.v_ids.1 = t;
        }

        if unsafe { edge.v_ids.0 != edge.v_ids.1 } {
            temp_edges.push(*edge);
        }
    }

    *edges = temp_edges;
}

// TODO: adapt to support VertexMesh::VertexImpl

pub fn simplify(vertex_reader: VertexReader, indices: &[u32], options: &Options) -> Vec<u32> {
    let mut vertices = vertex_reader.read_vertices();
    let mut indices = indices.to_vec();
    let mut edges = build_candidate_edges(vertices.len(), &indices);

    let mut vertex_triangle_counts = vec![0_u32; vertices.len()];

    for index in &indices {
        vertex_triangle_counts[*index as usize] += 1;
    }

    let mut collapse_positions = vec![na::Vector3::<f32>::default(); edges.len()];
    let mut iteration_count = 0;

    while (indices.len() / 3 >= options.target_triangle_count) && (iteration_count < options.max_iterations) {
        let mut collapse_edge_ids = vec![u32::MAX; vertices.len()];
        let mut collapse_target = vec![u32::MAX; vertices.len()];

        let valid_collapses = find_valid_collapses(
            &vertices,
            &vertex_triangle_counts,
            &edges,
            &options,
            &mut collapse_positions,
            &mut collapse_edge_ids,
        );
        if valid_collapses.is_empty() {
            break;
        }

        collapse_edges(
            &valid_collapses,
            &edges,
            &mut collapse_edge_ids,
            &mut collapse_target,
            &mut collapse_positions,
            &mut vertices,
        );
        remove_triangles(
            &vertices,
            &mut collapse_target,
            &mut indices,
            &mut vertex_triangle_counts,
        );
        remove_edges(&collapse_target, &mut edges);

        iteration_count += 1;
    }

    indices
}
