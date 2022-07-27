use crate::game::overworld;
use crate::game::overworld::structure::Structure;
use crate::game::overworld::Overworld;
use engine::utils::white_noise::WhiteNoise;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3};
use noise;
use noise::{NoiseFn, Seedable};
use rand::Rng;

pub const MIN_RADIUS: u64 = 2_048;
pub const MAX_RADIUS: u64 = 100_000;

pub struct World {
    seed: u64,
    grad_noise: noise::SuperSimplex,
    size: u64,
}

fn sample_world_size(rng: &mut impl Rng) -> u64 {
    const AVG_R: u64 = (MIN_RADIUS + MAX_RADIUS) / 2;
    const R_HALF_DIST: f64 = ((MAX_RADIUS - MIN_RADIUS) / 2) as f64;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_HALF_DIST).clamp(-R_HALF_DIST, R_HALF_DIST) as u64
}

impl World {
    pub fn new(seed: u64) -> World {
        World {
            seed,
            grad_noise: noise::SuperSimplex::new().set_seed((seed % (u32::MAX as u64)) as u32),
            size: sample_world_size(&mut WhiteNoise::<u64>::new(seed).state().rng()),
        }
    }

    pub fn is_land(&self, pos: I64Vec3) -> bool {
        // TODO: check also for rivers
        let mut pos: DVec3 = glm::convert(pos);
        pos.add_scalar_mut(0.5);
        self.grad_noise.get([pos.x, pos.y, pos.z]) > 0.0 // TODO
    }

    pub fn find_land(&self) -> I64Vec3 {
        todo!()
    }
}

fn gen_fn(_structure: &Structure, _overworld: &Overworld, _cluster_pos: I64Vec3) -> Option<I64Vec3> {
    todo!()
}
