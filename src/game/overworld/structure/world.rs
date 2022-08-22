pub mod biome;

use crate::game::overworld;
use crate::game::overworld::structure::Structure;
use crate::game::overworld::Overworld;
use engine::utils::voronoi_noise::VoronoiNoise3D;
use engine::utils::white_noise::WhiteNoise;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3};
use noise;
use noise::{NoiseFn, Seedable};
use rand::Rng;
use std::num::Wrapping;
use std::sync::Arc;

use crate::game::registry::Registry;
pub use biome::{Biome, BiomeSize};

pub const MIN_RADIUS: u64 = 2_048;
pub const MAX_RADIUS: u64 = 100_000;

pub struct BiomePartitionNoise {
    voronoi: VoronoiNoise3D,
    warp: [noise::SuperSimplex; 3],
}

pub struct World {
    seed: Wrapping<u64>,
    grad_noise: noise::SuperSimplex,
    biome_partition_noises: Vec<BiomePartitionNoise>,
    white_noise: WhiteNoise,
    size: u64,
    registry: Arc<Registry>,
}

fn sample_world_size(rng: &mut impl Rng) -> u64 {
    const AVG_R: u64 = (MIN_RADIUS + MAX_RADIUS) / 2;
    const R_HALF_DIST: f64 = ((MAX_RADIUS - MIN_RADIUS) / 2) as f64;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_HALF_DIST).clamp(-R_HALF_DIST, R_HALF_DIST) as u64
}

impl World {
    pub fn new(seed: u64, registry: Arc<Registry>) -> World {
        let seed = Wrapping(seed);
        let seed32 = Wrapping((seed % Wrapping(u32::MAX as u64)).0 as u32);

        let biome_partition_noises: Vec<_> = (0..(BiomeSize::MAX as u8 - BiomeSize::MIN as u8 + 1))
            .map(|i| BiomePartitionNoise {
                voronoi: VoronoiNoise3D::new()
                    .set_seed((seed + Wrapping(0x472935762) + Wrapping(i as u64)).0),
                warp: [
                    noise::SuperSimplex::new()
                        .set_seed((seed32 + Wrapping(i as u32) + Wrapping(0x93151_u32)).0),
                    noise::SuperSimplex::new()
                        .set_seed((seed32 + Wrapping(i as u32) + Wrapping(0x93152_u32)).0),
                    noise::SuperSimplex::new()
                        .set_seed((seed32 + Wrapping(i as u32) + Wrapping(0x93153_u32)).0),
                ],
            })
            .collect();

        World {
            seed,
            grad_noise: noise::SuperSimplex::new().set_seed(seed32.0),
            biome_partition_noises,
            white_noise: WhiteNoise::new(seed.0),
            size: sample_world_size(&mut WhiteNoise::new(seed.0).state().rng()),
            registry,
        }
    }

    /// May panic when there are no biomes of certain biome size.
    pub fn biome_at(&self, pos: I64Vec3) -> (&Biome, DVec3) {
        let pos_d: DVec3 = glm::convert(pos);
        let biomes = self.registry.biomes();

        // Note: using .rev() to iterate sizes from biggest to lowest
        for i in ((BiomeSize::MIN as u8)..=(BiomeSize::MAX as u8)).rev() {
            let noise = &self.biome_partition_noises[(i - BiomeSize::MIN as u8) as usize];

            let freq = 2.0_f64.powi(i as i32);
            let pos_df = pos_d / freq;

            let warp_x = noise.warp[0].get([pos_df.x, pos_df.y, pos_df.z]);
            let warp_y = noise.warp[1].get([pos_df.x, pos_df.y, pos_df.z]);
            let warp_z = noise.warp[2].get([pos_df.x, pos_df.y, pos_df.z]);
            let warp = (glm::vec3(warp_x, warp_y, warp_z) * 2.0).add_scalar(-1.0) * 0.1;

            // TODO: Optimize this by caching

            let (pivot, _) = noise.voronoi.sample(pos_df + warp);

            let mut rng = self
                .white_noise
                .state()
                .next(pivot.x)
                .next(pivot.y)
                .next(pivot.z)
                .rng();
            let biome_is_present = rng.gen::<bool>();

            // Also check for minimum-sized biome so there is always a biome present at any point
            if biome_is_present || (i == BiomeSize::MIN as u8) {
                // There is a biome at this size, select one
                let size = BiomeSize::from_level(i);
                let biomes = &biomes[&size];
                let biome_idx = rng.gen_range(0..biomes.len());

                return (&biomes[biome_idx], pivot);
            }
        }

        unreachable!()
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
