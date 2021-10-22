use crate::game::overworld;
use crate::game::overworld::structure::Structure;
use crate::game::overworld::Overworld;
use crate::utils::noise::ParamNoise;
use crate::utils::value_noise::ValueNoise;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3};
use noise::Seedable;

pub struct World {
    seed: u64,
    grad_noise: noise::SuperSimplex,
    size: u64,
}

impl World {
    pub fn new(seed: u64) -> World {
        World {
            seed,
            grad_noise: noise::SuperSimplex::new().set_seed((seed % (u32::MAX as u64)) as u32),
            size: overworld::sample_world_size(&mut ValueNoise::<u64>::new(seed).state().rng()),
        }
    }

    pub fn is_land(&self, pos: I64Vec3) -> bool {
        // TODO: check also for rivers
        let pos: DVec3 = glm::convert(pos);
        self.grad_noise.sample(pos.add_scalar(0.5), 1.0, 1.0, 0.5) > 0.5 // TODO
    }

    pub fn find_land(&self) -> I64Vec3 {
        todo!()
    }
}

fn gen_fn(_structure: &Structure, _overworld: &Overworld, _cluster_pos: I64Vec3) -> Option<I64Vec3> {
    todo!()
}
