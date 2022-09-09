pub mod biome;

use crate::game::overworld;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::generator::{OverworldGenerator, StructureCache};
use crate::game::overworld::structure::Structure;
use crate::game::overworld::Overworld;
use crate::game::registry::Registry;
use engine::utils::voronoi_noise::VoronoiNoise2D;
use engine::utils::white_noise::WhiteNoise;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec2, DVec3, I64Vec2, I64Vec3};
use noise;
use noise::{NoiseFn, Seedable};
use once_cell::sync::OnceCell;
use overworld::cluster;
use rand::Rng;
use rstar::{Envelope, Point, RTree};
use smallvec::SmallVec;
use std::any::Any;
use std::mem;
use std::num::Wrapping;
use std::ops::RangeInclusive;
use std::sync::Arc;

use crate::game::overworld::structure::world::biome::{MeanHumidity, MeanTemperature};
pub use biome::Biome;
use engine::utils::noise::{HybridNoise, ParamNoise};
use engine::utils::{ConcurrentCache, ConcurrentCacheImpl};

pub const MIN_RADIUS: u64 = 2_048;
pub const MAX_RADIUS: u64 = 100_000;

pub struct BiomePartitionNoise {
    voronoi: VoronoiNoise2D,
    warp: [noise::SuperSimplex; 2],
    white_noise: WhiteNoise,
}

fn sample_world_size(rng: &mut impl Rng) -> u64 {
    const AVG_R: u64 = (MIN_RADIUS + MAX_RADIUS) / 2;
    const R_HALF_DIST: f64 = ((MAX_RADIUS - MIN_RADIUS) / 2) as f64;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_HALF_DIST).clamp(-R_HALF_DIST, R_HALF_DIST) as u64
}

/// Gradually localizes smooth blobs into squares.
/// `value` and `midpoint` are in range [0, 1]; `strength`: [0, +inf]
fn localize(mut value: f32, midpoint: f32, strength: f32) -> f32 {
    let hp = if value > midpoint {
        1.0 / (1.0 + 2.0 * (value - midpoint) * strength)
    } else {
        1.0 + 2.0 * (midpoint - value) * strength
    };

    value.powf(hp)
}

pub struct ClimateRange {
    pub biome_id: u32,
    temp: RangeInclusive<f32>,
    humidity: RangeInclusive<f32>,
    altitude: RangeInclusive<f32>,
}

impl ClimateRange {
    pub fn new(
        biome_id: u32,
        temp: RangeInclusive<MeanTemperature>,
        humidity: RangeInclusive<MeanHumidity>,
        altitude: RangeInclusive<f32>,
    ) -> Self {
        Self {
            biome_id,
            temp: (*temp.start() as i32 as f32)..=(*temp.end() as i32 as f32),
            humidity: (*humidity.start() as i32 as f32)..=(*humidity.end() as i32 as f32),
            altitude,
        }
    }
}

impl rstar::RTreeObject for ClimateRange {
    type Envelope = rstar::AABB<[f32; 3]>;

    fn envelope(&self) -> Self::Envelope {
        rstar::AABB::from_corners(
            [*self.temp.start(), *self.humidity.start(), *self.altitude.start()],
            [*self.temp.end(), *self.humidity.end(), *self.altitude.end()],
        )
    }
}

impl rstar::PointDistance for ClimateRange {
    fn distance_2(
        &self,
        point: &<Self::Envelope as Envelope>::Point,
    ) -> <<Self::Envelope as Envelope>::Point as Point>::Scalar {
        let temp_mid = (self.temp.start() + self.temp.end()) / 2.0;
        let humidity_mid = (self.humidity.start() + self.humidity.end()) / 2.0;
        let altitude_mid = (self.altitude.start() + self.altitude.end()) / 2.0;

        (temp_mid - point[0]).powi(2) + (humidity_mid - point[1]).powi(2) + (altitude_mid - point[2]).powi(2)
    }

    fn contains_point(&self, point: &<Self::Envelope as Envelope>::Point) -> bool {
        self.temp.contains(&point[0])
            && self.humidity.contains(&point[1])
            && self.altitude.contains(&point[2])
    }
}

#[derive(Clone)]
struct ClusterXZCache {
    heights: Arc<[[f32; cluster::SIZE]; cluster::SIZE]>,
    biomes: Arc<[[u32; cluster::SIZE]; cluster::SIZE]>,
}

// Max 8192 2D clusters in cache
const MAX_CLUSTER_BIOME_MAPS: usize = 2048;
const MAX_GEN_HEIGHT: f32 = 500.0;

const FLATNESS_FREQ: f64 = 1.0 / 50.0;
const HILLS_FREQ: f64 = 1.0 / 100.0;
const HILLS_MASK_FREQ: f64 = 1.0 / 400.0;
const MOUNTAINS_FREQ: f64 = 1.0 / 1000.0;
const MOUNTAINS_MASK_FREQ: f64 = 1.0 / 700.0;
const MOUNTAINS_RIDGES_FREQ: f64 = 1.0 / 700.0;
const OCEANS_FREQ: f64 = 1.0 / 1400.0;
const OCEANS_LOC_FREQ: f64 = 1.0 / 700.0;

const FLATNESS_MAX_HEIGHT: f32 = 10.0;
const HILLS_MAX_HEIGHT: f32 = 60.0;
const MOUNTAINS_MAX_HEIGHT: f32 = 500.0;
const MAX_ALTITUDE: f32 = FLATNESS_MAX_HEIGHT + HILLS_MAX_HEIGHT + MOUNTAINS_MAX_HEIGHT;

pub struct WorldState {
    registry: Arc<Registry>,
    temperature_noise: ParamNoise<2, noise::SuperSimplex>,
    humidity_noise: ParamNoise<2, noise::SuperSimplex>,
    biome_partition_noise: BiomePartitionNoise,
    biomes_by_climate: RTree<ClimateRange>,
    cluster_xz_caches: ConcurrentCache<I64Vec2, ClusterXZCache>,

    flat_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    hills_noise: HybridNoise<2, 4, noise::SuperSimplex>,
    hills_mask_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    mountains_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    mountains_mask_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    mountains_midpoint_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    mountains_ridges_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    mountains_power_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    oceans_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    oceans_loc_noise: HybridNoise<2, 1, noise::SuperSimplex>,
    oceans_mask_noise: HybridNoise<2, 1, noise::SuperSimplex>,
}

impl WorldState {
    pub fn new(seed: u64, registry: &Arc<Registry>) -> Self {
        let white_noise = WhiteNoise::new(seed);
        let mut seed_gen = white_noise.state().rng();

        let biome_partition_noise = BiomePartitionNoise {
            voronoi: VoronoiNoise2D::new().set_seed(seed_gen.gen::<u64>()),
            warp: [
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
            ],
            white_noise: WhiteNoise::new(seed_gen.gen::<u64>()),
        };

        let biome_climate_ranges = registry.biomes().iter().enumerate().map(|(i, b)| {
            ClimateRange::new(i as u32, b.temp_range(), b.humidity_range(), b.altitude_range())
        });

        let biomes_by_climate = RTree::bulk_load(biome_climate_ranges.collect());

        if cfg!(debug_assertions) {
            // Assert the full climate range to be complete
            for t in (MeanTemperature::MIN as i32)..=(MeanTemperature::MAX as i32) {
                let t_factor = (t - MeanTemperature::MIN as i32) as f32 / MeanTemperature::SPREAD;
                let max_h =
                    ((t_factor * MeanHumidity::MAX as i32 as f32) as i32).max(MeanHumidity::H12 as i32);

                for h in (MeanHumidity::MIN as i32)..=max_h {
                    for a in -10..=10 {
                        if biomes_by_climate
                            .locate_at_point(&[t as f32, h as f32, a as f32 / 10.0])
                            .is_none()
                        {
                            panic!("No biome at T{} H{} climate!", t, h);
                        }
                    }
                }
            }
        }

        Self {
            registry: Arc::clone(registry),
            temperature_noise: ParamNoise::new(
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
                &[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            ),
            humidity_noise: ParamNoise::new(
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
                &[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            ),
            biome_partition_noise,
            biomes_by_climate,
            cluster_xz_caches: ConcurrentCache::new(MAX_CLUSTER_BIOME_MAPS),
            flat_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
            hills_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
            hills_mask_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
            mountains_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
            mountains_mask_noise: HybridNoise::new(
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
            ),
            mountains_midpoint_noise: HybridNoise::new(
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
            ),
            mountains_ridges_noise: HybridNoise::new(
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
            ),
            mountains_power_noise: HybridNoise::new(
                noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>()),
            ),
            oceans_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
            oceans_loc_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
            oceans_mask_noise: HybridNoise::new(noise::SuperSimplex::new().set_seed(seed_gen.gen::<u32>())),
        }
    }

    pub fn biomes_by_climate(&self) -> &RTree<ClimateRange> {
        &self.biomes_by_climate
    }

    pub fn sample_raw_temperature(&self, pos: I64Vec2) -> f32 {
        let pos_xz: DVec2 = glm::convert(pos);
        let value = self.temperature_noise.sample(pos_xz / 700.0);
        value as f32 * 0.5 + 0.5
    }

    pub fn sample_raw_humidity(&self, pos: I64Vec2) -> f32 {
        let pos_xz: DVec2 = glm::convert(pos);
        let value = self.humidity_noise.sample(pos_xz / 400.0);
        value as f32 * 0.5 + 0.5
    }

    fn calc_height_at(&self, pos: I64Vec2) -> f32 {
        let pos_d: DVec2 = glm::convert(pos);

        let mut flat_height = self.flat_noise.sample(pos_d * FLATNESS_FREQ, 0.5) as f32;
        flat_height = (flat_height * 0.5 + 0.5) * FLATNESS_MAX_HEIGHT;

        let mut hills_height = self.hills_noise.sample(pos_d * HILLS_FREQ, 0.5) as f32;
        hills_height = (hills_height * 0.5 + 0.5) * HILLS_MAX_HEIGHT;

        let mut hills_mask = self.hills_mask_noise.sample(pos_d * HILLS_MASK_FREQ, 0.5) as f32;
        hills_mask = hills_mask * 0.5 + 0.5;
        hills_mask = localize(hills_mask, 0.54, 12.0);

        let mut mountains_height = self.mountains_noise.sample(pos_d * MOUNTAINS_FREQ, 0.5) as f32;
        mountains_height = mountains_height * 0.5 + 0.5;

        let mut mountains_mask = self.mountains_mask_noise.sample(pos_d * MOUNTAINS_MASK_FREQ, 0.5) as f32;
        mountains_mask = mountains_mask * 0.5 + 0.5;
        mountains_mask = localize(mountains_mask, 0.5, 4.);

        let mut mountains_mp = self
            .mountains_midpoint_noise
            .sample(pos_d * MOUNTAINS_MASK_FREQ, 0.5) as f32;
        mountains_mp = (mountains_mp * 0.5 + 0.5) * 0.4 + 0.3;

        let mut mountains_ridges = self
            .mountains_ridges_noise
            .sample(pos_d * MOUNTAINS_RIDGES_FREQ, 0.5) as f32;
        mountains_ridges = (1.0 - mountains_ridges.abs()).powf(4.0);

        let mut mountains_pow = self
            .mountains_power_noise
            .sample(pos_d * MOUNTAINS_RIDGES_FREQ, 0.5) as f32;
        mountains_pow = (mountains_pow * 0.5 + 0.5);

        mountains_height = localize(mountains_height, mountains_mp, mountains_mask * 200.0);
        mountains_height *= glm::mix_scalar(1., mountains_ridges, mountains_pow);
        mountains_height *= MOUNTAINS_MAX_HEIGHT;

        let mut oceans_loc = self.oceans_loc_noise.sample(pos_d * OCEANS_LOC_FREQ, 0.5) as f32;
        oceans_loc = oceans_loc * 0.5 + 0.5;

        let mut oceans = self.oceans_noise.sample(pos_d * OCEANS_FREQ, 0.5) as f32;
        oceans = oceans * 0.5 + 0.5;
        oceans = localize(oceans, 0.5, 2.0_f32.powf(oceans_loc * 10.0));
        oceans = -1.0 + oceans;
        oceans *= MAX_ALTITUDE;

        let mut oceans_mask = self.oceans_mask_noise.sample(pos_d * OCEANS_LOC_FREQ, 0.5) as f32;
        oceans_mask = oceans_mask * 0.5 + 0.5;
        oceans_mask = localize(oceans_mask, 0.5, 4.);

        let final_height = flat_height
            + hills_height * hills_mask
            + mountains_height * mountains_mask
            + oceans * oceans_mask;

        final_height.clamp(-MAX_ALTITUDE, MAX_ALTITUDE)
    }

    pub fn select_biome_idx(&self, pos: I64Vec2, max: usize) -> usize {
        let freq = 1.0 / 150.0;
        let pos_d = glm::convert::<_, DVec2>(pos) * freq;

        let warp_x = self.biome_partition_noise.warp[0].get([pos_d.x, pos_d.y]);
        let warp_y = self.biome_partition_noise.warp[1].get([pos_d.x, pos_d.y]);
        let warp = (glm::vec2(warp_x, warp_y) * 2.0).add_scalar(-1.0) * 0.1;

        let (pivot, _) = self.biome_partition_noise.voronoi.sample(pos_d + warp);

        self.biome_partition_noise
            .white_noise
            .state()
            .next(pivot.x as i64)
            .next(pivot.y as i64)
            .rng()
            .gen_range(0..max)
    }

    fn calc_biome_at(&self, pos: I64Vec2, height: f32) -> u32 {
        let mut raw_temp = self.sample_raw_temperature(pos);
        let mut raw_humidity = self.sample_raw_humidity(pos);
        let height = self.calc_height_at(pos); // TODO: CACHE THIS
        let raw_altitude = height / MAX_ALTITUDE;

        if !(-1.0..=1.0).contains(&raw_altitude) {
            panic!("GOV {}", raw_altitude);
        }

        if raw_humidity > raw_temp {
            // When the temperature is low the humidity must also be low,
            // hence mirror climate parameters about triangle diagonal
            mem::swap(&mut raw_temp, &mut raw_humidity);
        }

        let temp = MeanTemperature::MIN as i32 as f32 + (raw_temp * MeanTemperature::SPREAD).ceil();
        let humidity = MeanHumidity::MIN as i32 as f32 + (raw_humidity * MeanHumidity::SPREAD);

        let biomes: SmallVec<[u32; 16]> = self
            .biomes_by_climate
            .locate_all_at_point(&[temp, humidity, raw_altitude])
            .map(|v| v.biome_id)
            .collect();

        if biomes.len() >= 2 {
            // Select biome using voronoi noise
            let idx = self.select_biome_idx(pos, biomes.len());
            biomes[idx]
        } else {
            biomes[0]
        }
    }

    fn build_cluster_xz_cache(&self, cluster_pos: I64Vec2) -> ClusterXZCache {
        let mut height_map = Arc::<[[f32; cluster::SIZE]; cluster::SIZE]>::new(Default::default());
        let height_map_mut = Arc::get_mut(&mut height_map).unwrap();

        let mut biome_map = Arc::<[[u32; cluster::SIZE]; cluster::SIZE]>::new(Default::default());
        let biome_map_mut = Arc::get_mut(&mut biome_map).unwrap();

        for x in 0..cluster::SIZE {
            for z in 0..cluster::SIZE {
                let pos = cluster_pos + I64Vec2::new(x as i64, z as i64);
                let height = self.calc_height_at(pos);

                height_map_mut[x][z] = height;
                biome_map_mut[x][z] = self.calc_biome_at(pos, height);
            }
        }

        ClusterXZCache {
            heights: height_map,
            biomes: biome_map,
        }
    }

    pub fn biome_2d_at(&self, pos: I64Vec2) -> u32 {
        let cluster_pos = pos.map(|v| v.div_euclid(cluster::SIZE as i64) * cluster::SIZE as i64);
        let rel_pos = pos.map(|v| v.rem_euclid(cluster::SIZE as i64));

        let cache = self
            .cluster_xz_caches
            .get_with(cluster_pos, || self.build_cluster_xz_cache(cluster_pos));

        cache.biomes[rel_pos.x as usize][rel_pos.y as usize]
    }

    pub fn is_land(&self, pos: I64Vec3) -> bool {
        todo!()
    }

    pub fn find_land(&self) -> I64Vec3 {
        todo!()
    }
}

impl StructureCache for WorldState {
    fn size(&self) -> u32 {
        let cache_size = cluster::SIZE * cluster::SIZE * mem::size_of::<u32>();
        (self.cluster_xz_caches.weighted_size() * cache_size as u64 / 1024) as u32
    }
}

pub fn gen_fn(
    structure: &Structure,
    generator: &OverworldGenerator,
    structure_seed: u64,
    cluster_pos: I64Vec3,
    cluster: &mut Cluster,
    structure_cache: Arc<OnceCell<Box<dyn StructureCache>>>,
) {
    let cache = structure_cache.get_or_init(|| {
        Box::new(WorldState::new(
            structure_seed,
            generator.main_registry().registry(),
        ))
    });

    for x in 0..cluster::SIZE {
        for y in 0..cluster::SIZE {
            for z in 0..cluster::SIZE {}
        }
    }
}
