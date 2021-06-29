pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster;
mod generator;
pub mod streamer;
pub mod structure;
pub mod textured_block_model;

use nalgebra_glm::I64Vec3;
use rand::Rng;

use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::structure::Structure;
use crate::game::registry::GameRegistry;
use crate::utils::value_noise::ValueNoise;
use crate::utils::HashMap;
use nalgebra_glm as glm;
use std::iter::Filter;
use std::slice::Iter;
use std::sync::Arc;

// TODO Main world - 'The Origin'

const MIN_WORLD_RADIUS: f64 = 2_048.0;
const MAX_WORLD_RADIUS: f64 = 32_000_000.0;
const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

pub const MAX_LOD: usize = 24;

fn sample_world_size(rng: &mut impl rand::Rng) -> f64 {
    const AVG_R: f64 = (MIN_WORLD_RADIUS + MAX_WORLD_RADIUS) / 2.0;
    const R_HALF_DIST: f64 = (MAX_WORLD_RADIUS - MIN_WORLD_RADIUS) / 2.0;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_HALF_DIST).clamp(-R_HALF_DIST, R_HALF_DIST)
}

pub struct Overworld {
    seed: u64,
    loaded_clusters: [HashMap<I64Vec3, Cluster>; MAX_LOD],
    registry: Arc<GameRegistry>,
    value_noise: ValueNoise<u64>,
}

impl Overworld {
    pub fn new(registry: &Arc<GameRegistry>, seed: u64) -> Overworld {
        Overworld {
            seed,
            loaded_clusters: Default::default(),
            registry: Arc::clone(registry),
            value_noise: ValueNoise::new(seed),
        }
    }

    fn gen_spawn_point(&self) -> I64Vec3 {
        todo!()
    }

    pub fn load_cluster(&self) {
        todo!()
    }
}

// Overworld creation
// 1. Determine player position
//   1. Find a closest world to 0-coordinate. Choose a random reasonable position X in the world.
//   2. Find non-flooded-with-liquid cluster around X.
//   3. Choose a random reasonable player position in this cluster.
// 2. Start generating the overworld
