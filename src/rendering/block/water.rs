use algebra_glm::{DVec3, Vec3};

use crate::overworld::block_model::{BlockModel, Quad};
use crate::overworld::textured_block_model::{QuadMaterial, TexturedBlockModel};
use crate::physics::aabb::AABB;
use crate::registry::Registry;

fn construct(heights: [f32; 4]) -> BlockModel {
    let p000 = Vec3::new(0.0, 0.0, 0.0);
    let p001 = Vec3::new(0.0, 0.0, 1.0);
    let p100 = Vec3::new(1.0, 0.0, 0.0);
    let p101 = Vec3::new(1.0, 0.0, 1.0);

    let p010 = Vec3::new(0.0, heights[0], 0.0);
    let p011 = Vec3::new(0.0, heights[1], 1.0);
    let p110 = Vec3::new(1.0, heights[2], 0.0);
    let p111 = Vec3::new(1.0, heights[3], 1.0);

    BlockModel::new(
        &[
            // Z
            Quad::new([p110, p100, p010, p000]),
            Quad::new([p011, p001, p111, p101]),
            // Y
            Quad::new([p000, p100, p001, p101]),
            Quad::new([p011, p111, p010, p110]),
            // X
            Quad::new([p010, p000, p011, p001]),
            Quad::new([p111, p101, p110, p100]),
        ],
        &[AABB::new(
            DVec3::from_element(0.0),
            DVec3::from_element(*heights.iter().max_by(|a, b| a.total_cmp(b)).unwrap() as f64),
        )],
    )
}

pub fn v4d_to_idx(vals: [usize; 4], size: usize) -> usize {
    let size_3d = size * size * size;
    let size_2d = size * size;

    vals[0] * size_3d + vals[1] * size_2d + vals[2] * size + vals[3]
}

pub fn idx_to_4d(idx: usize, size: usize) -> [usize; 4] {
    let size_3d = size * size * size;
    let size_2d = size * size;

    let idx_3d = idx % size_3d;
    let idx_2d = idx_3d % size_2d;
    let idx_1d = idx_3d % size;

    [idx / size_3d, idx_3d / size_2d, idx_2d / size, idx_1d]
}

pub fn gen_blocks(reg: &Registry, material_id: u16, levels: usize) -> Vec<TexturedBlockModel> {
    let s = 1.0 / levels as f32;
    let n_states = levels.pow(4);

    (0..n_states)
        .map(|i| {
            let vals = idx_to_4d(i, levels);
            let model = construct([
                vals[0] as f32 * s,
                vals[1] as f32 * s,
                vals[2] as f32 * s,
                vals[3] as f32 * s,
            ]);

            TexturedBlockModel::new(&model, &[QuadMaterial::new(material_id); 6], reg)
        })
        .collect::<Vec<_>>()
}
