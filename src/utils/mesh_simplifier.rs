use crate::renderer::vertex_mesh::{VertexImpl, VertexNormalImpl};
use crate::utils::qef;
use nalgebra as na;

const COLLAPSE_MAX_DEGREE: u32 = 16;

#[repr(C)]
#[derive(Copy, Clone)]
union Edge {
    id: u64,
    v_ids: (u32, u32),
}

pub struct Options {
    /// Each iteration involves selecting a fraction of the edges at random as possible
    /// candidates for collapsing. There is likely a sweet spot here trading off against number
    /// of edges processed vs number of invalid collapses generated due to collisions
    /// (the more edges that are processed the higher the chance of collisions happening)
    edge_fraction: f32,
    /// Stop simplfying after a given number of iterations
    max_iterations: usize,
    /// And/or stop simplifying when we've reached a threshold of the input triangles
    target_triangle_count: usize,
    /// Useful for controlling how uniform the mesh is (or isn't)
    max_edge_size: f32,
    /// The maximum allowed error when collapsing an edge (error is calculated as 1.0/qef_error)
    max_error: f32,
    /// If the mesh has sharp edges this can used to prevent collapses which would otherwise be used
    min_angle_cosine: f32,
}

impl Options {
    pub fn new(
        edge_fraction: f32,
        max_iterations: usize,
        target_triangle_count: usize,
        max_edge_size: f32,
        max_error: f32,
        min_angle_cosine: f32,
    ) -> Options {
        Options {
            edge_fraction,
            max_iterations,
            target_triangle_count,
            max_edge_size,
            max_error,
            min_angle_cosine,
        }
    }
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

    if edges.is_empty() {
        return edges;
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

fn find_valid_collapses<T>(
    vertices: &[T],
    vertex_triangle_counts: &[u32],
    edges: &[Edge],
    options: &Options,
    collapse_positions: &mut [na::Vector3<f32>],
    collapse_normals: &mut [na::Vector3<f32>],
) -> (Vec<u32>, Vec<u32>)
where
    T: VertexImpl + VertexNormalImpl,
{
    let step = ((1.0 / options.edge_fraction).round() as usize).max(1);
    let target_valid_edges = (edges.len() as f64 * options.edge_fraction as f64) as usize;

    if target_valid_edges == 0 {
        return (vec![], vec![u32::MAX; vertices.len()]);
    }

    let mut valid_collapses = Vec::<u32>::with_capacity(edges.len());
    let mut collapse_edge_ids = vec![u32::MAX; vertices.len()];
    let mut min_edge_cost = vec![f32::MAX; vertices.len()];

    for i in 0..step {
        let mut state = false;

        for j in (i..edges.len()).step_by(step) {
            if valid_collapses.len() >= target_valid_edges {
                state = true;
                break;
            }

            let edge = &edges[j as usize];
            let edge_i0 = unsafe { edge.v_ids.0 } as usize;
            let edge_i1 = unsafe { edge.v_ids.1 } as usize;
            let v0 = &vertices[edge_i0];
            let v1 = &vertices[edge_i1];

            if v0.normal().dot(v1.normal()) < options.min_angle_cosine {
                continue;
            }

            if (v1.position() - v0.position()).magnitude() > options.max_edge_size {
                continue;
            }

            let degree = vertex_triangle_counts[edge_i0] + vertex_triangle_counts[edge_i1];
            if degree > COLLAPSE_MAX_DEGREE {
                continue;
            }

            let (solved_pos, mut error) =
                qef::solve(&[(*v0.position(), *v0.normal()), (*v1.position(), *v1.normal())]);
            if error > 0.0 {
                error = 1.0 / error;
            }

            // avoid vertices becoming a 'hub' for lots of edges by penalising collapses
            // which will lead to a vertex with degree > 10
            let penalty = 0.max(degree as i32 - 10) as f32;
            error += penalty * (options.max_error * 0.1);

            if error > options.max_error {
                continue;
            }

            valid_collapses.push(j as u32);

            collapse_positions[j] = solved_pos; //(v0.position() + v1.position()) * 0.5;
            collapse_normals[j] = (v0.normal() + v1.normal()) * 0.5;

            if error < min_edge_cost[edge_i0] {
                min_edge_cost[edge_i0] = error;
                collapse_edge_ids[edge_i0] = j as u32;
            }
            if error < min_edge_cost[edge_i1] {
                min_edge_cost[edge_i1] = error;
                collapse_edge_ids[edge_i1] = j as u32;
            }
        }

        if state {
            break;
        }
    }

    (valid_collapses, collapse_edge_ids)
}

fn collapse_edges<T>(
    valid_collapses: &[u32],
    edges: &[Edge],
    collapse_edge_ids: &[u32],
    collapse_positions: &[na::Vector3<f32>],
    collapse_normals: &[na::Vector3<f32>],
    vertices: &mut [T],
) -> Vec<u32>
where
    T: VertexImpl + VertexNormalImpl,
{
    let mut collapse_target = vec![u32::MAX; vertices.len()];

    for &i in valid_collapses.iter() {
        let edge = &edges[i as usize];
        let edge_i0 = unsafe { edge.v_ids.0 } as usize;
        let edge_i1 = unsafe { edge.v_ids.1 } as usize;

        if (collapse_edge_ids[edge_i0] == i) && (collapse_edge_ids[edge_i1] == i) {
            collapse_target[edge_i1] = edge_i0 as u32;
            *vertices[edge_i0].position_mut() = collapse_positions[i as usize];
            *vertices[edge_i0].normal_mut() = collapse_normals[i as usize];
        }
    }

    collapse_target
}

fn remove_triangles(
    vertex_count: usize,
    collapse_target: &mut [u32],
    indices: &mut Vec<u32>,
    vertex_triangle_counts: &mut Vec<u32>,
) {
    let mut temp_indices = Vec::<u32>::with_capacity(indices.len());
    *vertex_triangle_counts = vec![0_u32; vertex_count];

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

fn compact_vertices<T>(vertices: &[T], indices: &mut [u32]) -> Vec<T>
where
    T: VertexImpl + Clone,
{
    let mut vertex_used = vec![false; vertices.len()];

    for index in indices.iter_mut() {
        vertex_used[*index as usize] = true;
    }

    let mut compacted_vertices = Vec::with_capacity(vertices.len());
    let mut remapped_indices = vec![u32::MAX; vertices.len()];

    for (i, vertex) in vertices.iter().enumerate() {
        if vertex_used[i] {
            remapped_indices[i] = compacted_vertices.len() as u32;
            compacted_vertices.push(vertex.clone());
        }
    }

    for index in indices.iter_mut() {
        *index = remapped_indices[*index as usize];
    }

    compacted_vertices
}

pub fn simplify<T>(vertices: &[T], indices: &[u32], options: &Options) -> (Vec<T>, Vec<u32>)
where
    T: VertexImpl + VertexNormalImpl + Clone,
{
    // Remove zero-area triangles
    let mut new_indices: Vec<u32> = Vec::with_capacity(indices.len());

    for i in (0..indices.len()).step_by(3) {
        let ind = &indices[i..(i + 3)];

        let v0 = &vertices[ind[0] as usize];
        let v1 = &vertices[ind[1] as usize];
        let v2 = &vertices[ind[2] as usize];

        if v0.position() == v1.position() || v0.position() == v2.position() || v1.position() == v2.position()
        {
            continue;
        }

        new_indices.extend(ind.iter());
    }

    let mut vertices = vertices.to_vec();
    // let mut pos_vertices: Vec<na::Vector3<f32>> = vertices.iter().map(|vertex| *vertex.position()).collect();
    let mut indices = new_indices;
    // let mut indices = indices.to_vec();
    let mut edges = build_candidate_edges(vertices.len(), &indices);

    let mut vertex_triangle_counts = vec![0_u32; vertices.len()];

    for index in &indices {
        vertex_triangle_counts[*index as usize] += 1;
    }

    let mut collapse_positions = vec![na::Vector3::<f32>::default(); edges.len()];
    let mut collapse_normals = vec![na::Vector3::<f32>::default(); edges.len()];
    let mut iteration_count = 0;

    while (indices.len() / 3 >= options.target_triangle_count) && (iteration_count < options.max_iterations) {
        let (valid_collapses, collapse_edge_ids) = find_valid_collapses(
            &vertices,
            &vertex_triangle_counts,
            &edges,
            &options,
            &mut collapse_positions,
            &mut collapse_normals,
        );
        if valid_collapses.is_empty() {
            break;
        }

        let mut collapse_target = collapse_edges(
            &valid_collapses,
            &edges,
            &collapse_edge_ids,
            &collapse_positions,
            &collapse_normals,
            &mut vertices,
        );
        remove_triangles(
            vertices.len(),
            &mut collapse_target,
            &mut indices,
            &mut vertex_triangle_counts,
        );
        remove_edges(&collapse_target, &mut edges);

        iteration_count += 1;
    }

    let vertices = compact_vertices(&mut vertices, &mut indices);

    (vertices, indices)
}
