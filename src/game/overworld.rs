use rand::Rng;

pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster;
mod generator;
pub mod streamer;
pub mod textured_block_model;
pub mod world_model;

const MIN_WORLD_RADIUS: f64 = 2_048.0;
const MAX_WORLD_RADIUS: f64 = 32_000_000.0;

fn sample_world_size(rng: &mut impl rand::Rng) -> f64 {
    const AVG_R: f64 = (MIN_WORLD_RADIUS + MAX_WORLD_RADIUS) / 2.0;
    const R_DIST: f64 = (MAX_WORLD_RADIUS - MIN_WORLD_RADIUS) / 2.0;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_DIST)
}

#[derive(Default)]
pub struct Overworld {
    seed: u64,
}

impl Overworld {
    pub fn new(seed: u64) -> Overworld {
        Overworld { seed }
    }
}

// Main world - 'The Origin'
