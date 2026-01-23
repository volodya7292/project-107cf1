pub mod biome;

use crate::overworld::facing::Facing;
use crate::overworld::generator::{OverworldGenerator, StructureCache};
use crate::overworld::light_state::LightLevel;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::{BlockPos, ClusterBlockPos};
use crate::overworld::raw_cluster::{BlockDataImpl, RawCluster};
use crate::overworld::structure::Structure;
use crate::overworld::structure::world::biome::{MeanHumidity, MeanTemperature};
use crate::registry::Registry;
use crate::utils::noise::{HybridNoise, ParamNoise};
use crate::utils::voronoi_noise::VoronoiNoise2D;
use crate::utils::white_noise::WhiteNoise;
pub use biome::Biome;
use bit_vec::BitVec;
use common::glm;
use common::glm::{DVec2, I64Vec2, I64Vec3};
use common::types::ConcurrentCache;
use common::types::ConcurrentCacheExt;
use noise;
use noise::{NoiseFn, Seedable};
use rand::Rng;
use rand_distr::num_traits::Zero;
use rstar::{Envelope, Point, RTree};
use smallvec::SmallVec;
use std::any::Any;
use std::collections::VecDeque;
use std::mem;
use std::ops::RangeInclusive;
use std::sync::{Arc, OnceLock};

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
fn localize(value: f32, midpoint: f32, strength: f32) -> f32 {
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
    heights: Arc<[[f32; RawCluster::SIZE]; RawCluster::SIZE]>,
    biomes: Arc<[[u32; RawCluster::SIZE]; RawCluster::SIZE]>,
}

// Max 2D cluster XZ->biome mappings in cache
const MAX_CLUSTER_BIOME_MAPS: usize = 2048;
const MAX_OCEAN_DEPTH: f32 = 256.0;
const MAX_SURFACE_HEIGHT: f32 = 256.0;
// Do not change, this depends on min/max values
const WATER_LEVEL_ALTITUDE: f32 = 0.0;
const GEN_ALTITUDE_RANGE: f32 = MAX_OCEAN_DEPTH + MAX_SURFACE_HEIGHT;

const FLATNESS_FREQ: f64 = 1.0 / 50.0;
const HILLS_FREQ: f64 = 1.0 / 100.0;
const HILLS_MASK_FREQ: f64 = 1.0 / 400.0;
const MOUNTAINS_FREQ: f64 = 1.0 / 1000.0;
const MOUNTAINS_MASK_FREQ: f64 = 1.0 / 700.0;
const MOUNTAINS_RIDGES_FREQ: f64 = 1.0 / 700.0;
const OCEANS_FREQ: f64 = 1.0 / 1400.0;
const OCEANS_LOC_FREQ: f64 = 1.0 / 700.0;

const FLATNESS_MAX_HEIGHT: f32 = 0.03 * MAX_SURFACE_HEIGHT; //8.0;
const HILLS_MAX_HEIGHT: f32 = 0.17 * MAX_SURFACE_HEIGHT; //60.0;
const MOUNTAINS_MAX_HEIGHT: f32 = 0.8 * MAX_SURFACE_HEIGHT; //400.0;

pub struct WorldState {
    registry: Arc<Registry>,
    temperature_noise: ParamNoise<2, noise::SuperSimplex>,
    humidity_noise: ParamNoise<2, noise::SuperSimplex>,
    biome_partition_noise: BiomePartitionNoise,
    biomes_by_climate: RTree<ClimateRange>,
    cluster_xz_caches: ConcurrentCache<I64Vec2, ClusterXZCache>,

    flat_noise: ParamNoise<2, noise::SuperSimplex>,
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
        let rng_state = white_noise.state();
        let mut seed_gen = rng_state.rng();

        let biome_partition_noise = BiomePartitionNoise {
            voronoi: VoronoiNoise2D::new().set_seed(seed_gen.random::<u64>()),
            warp: [
                noise::SuperSimplex::new(seed_gen.random::<u32>()),
                noise::SuperSimplex::new(seed_gen.random::<u32>()),
            ],
            white_noise: WhiteNoise::new(seed_gen.random::<u64>()),
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
                noise::SuperSimplex::new(seed_gen.random::<u32>()),
                &[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            ),
            humidity_noise: ParamNoise::new(
                noise::SuperSimplex::new(seed_gen.random::<u32>()),
                &[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
            ),
            biome_partition_noise,
            biomes_by_climate,
            cluster_xz_caches: ConcurrentCache::new(512),
            flat_noise: ParamNoise::new(
                noise::SuperSimplex::new(seed_gen.random::<u32>()),
                &[(1.0, 1.0), (2.0, 0.4), (4.0, 0.2)],
            ),
            hills_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            hills_mask_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            mountains_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            mountains_mask_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            mountains_midpoint_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            mountains_ridges_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            mountains_power_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            oceans_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            oceans_loc_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
            oceans_mask_noise: HybridNoise::new(noise::SuperSimplex::new(seed_gen.random::<u32>())),
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

    fn calc_altitude_at(&self, pos: I64Vec2) -> f32 {
        let pos_d: DVec2 = glm::convert(pos);

        let mut flat_height = self.flat_noise.sample(pos_d * FLATNESS_FREQ) as f32;
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
        mountains_pow = mountains_pow * 0.5 + 0.5;

        mountains_height = localize(mountains_height, mountains_mp, mountains_mask * 200.0);
        mountains_height *= glm::mix_scalar(1., mountains_ridges, mountains_pow);
        mountains_height *= MOUNTAINS_MAX_HEIGHT;

        let mut oceans_loc = self.oceans_loc_noise.sample(pos_d * OCEANS_LOC_FREQ, 0.5) as f32;
        oceans_loc = oceans_loc * 0.5 + 0.5;

        let mut oceans = self.oceans_noise.sample(pos_d * OCEANS_FREQ, 0.5) as f32;
        oceans = oceans * 0.5 + 0.5;
        oceans = localize(oceans, 0.5, 2.0_f32.powf(oceans_loc * 10.0));

        // let mut oceans_mask = self.oceans_mask_noise.sample(pos_d * OCEANS_LOC_FREQ, 0.5) as f32;
        // oceans_mask = oceans_mask * 0.5 + 0.5;
        // oceans_mask = localize(oceans_mask, 0.5, 4.);

        // Transform water level so that the MIN_GEN_ALTITUDE is at 0
        const WATER_LEVEL_ZO: f32 = 0.5 * GEN_ALTITUDE_RANGE + WATER_LEVEL_ALTITUDE;

        // Range [water level height; max altitude]
        let surface_height =
            WATER_LEVEL_ZO + flat_height + hills_height * hills_mask + mountains_height * mountains_mask;

        // Range [0; MAX_HEIGHT]
        // `oceans_mask` makes sure that when there's an ocean,
        // the height is at 0 when surface_height is at WATER_LEVEL_ZO;
        // similarly, when surface_height is max, the final_height is always < WATER_LEVEL_ZO.
        let final_height = surface_height * oceans;

        (final_height - 0.5 * GEN_ALTITUDE_RANGE).clamp(-MAX_OCEAN_DEPTH, MAX_SURFACE_HEIGHT)
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

    fn calc_biome_at(&self, pos: I64Vec2, altitude: f32) -> u32 {
        let mut raw_temp = self.sample_raw_temperature(pos);
        let mut raw_humidity = self.sample_raw_humidity(pos);
        let raw_altitude = ((altitude + MAX_OCEAN_DEPTH) / GEN_ALTITUDE_RANGE) * 2.0 - 1.0;

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
        let mut height_map = Arc::<[[f32; RawCluster::SIZE]; RawCluster::SIZE]>::new(Default::default());
        let height_map_mut = Arc::get_mut(&mut height_map).unwrap();

        let mut biome_map = Arc::<[[u32; RawCluster::SIZE]; RawCluster::SIZE]>::new(Default::default());
        let biome_map_mut = Arc::get_mut(&mut biome_map).unwrap();

        for x in 0..RawCluster::SIZE {
            for z in 0..RawCluster::SIZE {
                let pos = cluster_pos + I64Vec2::new(x as i64, z as i64);
                let altitude = self.calc_altitude_at(pos);

                height_map_mut[x][z] = altitude;
                biome_map_mut[x][z] = self.calc_biome_at(pos, altitude);
            }
        }

        ClusterXZCache {
            heights: height_map,
            biomes: biome_map,
        }
    }

    fn cluster_xz_cache_at(&self, cluster_pos: I64Vec2) -> ClusterXZCache {
        self.cluster_xz_caches
            .get_with(cluster_pos, || self.build_cluster_xz_cache(cluster_pos))
    }

    pub fn biome_2d_at(&self, pos: I64Vec2) -> u32 {
        let cluster_pos = pos.map(|v| v.div_euclid(RawCluster::SIZE as i64) * RawCluster::SIZE as i64);
        let rel_pos = pos.map(|v| v.rem_euclid(RawCluster::SIZE as i64));
        let cache = self.cluster_xz_cache_at(cluster_pos);
        cache.biomes[rel_pos.x as usize][rel_pos.y as usize]
    }

    pub fn is_land(&self, _pos: I64Vec3) -> bool {
        todo!()
    }

    pub fn find_land(&self, closest_to: I64Vec2) -> BlockPos {
        const SEARCH_DIAM: i64 = 32;
        const CLUSTER_CENTER: i64 = RawCluster::SIZE as i64 / 2;

        let cluster_closest_to = closest_to.map(|v| v.div_euclid(RawCluster::SIZE as i64));
        let mut queue = VecDeque::with_capacity((SEARCH_DIAM * SEARCH_DIAM) as usize);
        let mut traversed_nodes = BitVec::from_elem(queue.capacity(), false);

        {
            traversed_nodes.set(0, true);

            let block_pos = (cluster_closest_to * RawCluster::SIZE as i64).add_scalar(CLUSTER_CENTER);
            let altitude = self.calc_altitude_at(block_pos);

            if altitude >= 1.0 {
                return BlockPos::new(block_pos.x, altitude as i64 + 1, block_pos.y);
            }
        }

        queue.push_back(cluster_closest_to);

        // Breadth-first search
        while let Some(curr_pos) = queue.pop_front() {
            for dir in &Facing::XZ_DIRECTIONS {
                let dir: I64Vec2 = glm::convert(dir.xz());
                let next_pos = curr_pos + dir;

                let rel_pos = (next_pos - cluster_closest_to).add_scalar(SEARCH_DIAM / 2);
                if rel_pos.x < 0 || rel_pos.y < 0 || rel_pos.x >= SEARCH_DIAM || rel_pos.y >= SEARCH_DIAM {
                    continue;
                }

                let idx_1d = (rel_pos.y * SEARCH_DIAM + rel_pos.x) as usize;
                if traversed_nodes.get(idx_1d).unwrap() {
                    continue;
                }

                queue.push_back(next_pos);
                traversed_nodes.set(idx_1d, true);

                let block_pos = (next_pos * RawCluster::SIZE as i64).add_scalar(CLUSTER_CENTER);
                let altitude = self.calc_altitude_at(block_pos);

                if altitude >= 1.0 {
                    return BlockPos::new(block_pos.x, altitude as i64 + 1, block_pos.y);
                }
            }
        }

        let height = self.calc_altitude_at(closest_to);
        BlockPos::new(closest_to.x, height as i64 + 1, closest_to.y)
    }
}

impl StructureCache for WorldState {
    fn size(&self) -> u32 {
        let cache_size = RawCluster::SIZE * RawCluster::SIZE * mem::size_of::<u32>();
        (self.cluster_xz_caches.weighted_size() * cache_size as u64 / 1024) as u32
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub fn gen_fn(
    _structure: &Structure,
    generator: &OverworldGenerator,
    structure_seed: u64,
    cluster_origin: BlockPos,
    cluster: &mut RawCluster,
    state: Arc<OnceLock<Box<dyn StructureCache>>>,
) {
    let state = state
        .get_or_init(|| {
            Box::new(WorldState::new(
                structure_seed,
                generator.main_registry().registry(),
            ))
        })
        .as_any()
        .downcast_ref::<WorldState>()
        .unwrap();

    let registry = generator.main_registry();
    let xz_cache = state.cluster_xz_cache_at(cluster_origin.cluster_pos().get().xz());

    for x in 0..RawCluster::SIZE {
        for y in 0..RawCluster::SIZE {
            for z in 0..RawCluster::SIZE {
                let global_y = cluster_origin.0.y + y as i64;
                let altitude = xz_cache.heights[x][z];
                let pos = ClusterBlockPos::new(x, y, z);
                let mut data = cluster.get_mut(&pos);

                // May be not necessary
                // if global_y.rem_euclid(1024) == 0 {
                //     *data.light_source_type_mut() = LightType::Sky;
                //     *data.raw_light_source_mut() = LightLevel::MAX;
                // }

                if global_y <= altitude as i64 {
                    data.set(registry.block_test);
                } else {
                    data.set(registry.block_empty);

                    if global_y <= WATER_LEVEL_ALTITUDE as i64 {
                        *data.liquid_state_mut() = LiquidState::source(registry.liquid_water);

                        let liquid_depth = (WATER_LEVEL_ALTITUDE as i64 - global_y) as u8;
                        *data.sky_light_state_mut() = crate::sky_light_value_in_liquid(liquid_depth);
                    } else {
                        *data.sky_light_state_mut() = LightLevel::MAX;
                    }
                }
            }
        }
    }
}

pub fn spawn_point_fn(
    _structure: &Structure,
    generator: &OverworldGenerator,
    structure_seed: u64,
    state: Arc<OnceLock<Box<dyn StructureCache>>>,
) -> BlockPos {
    let state = state
        .get_or_init(|| {
            Box::new(WorldState::new(
                structure_seed,
                generator.main_registry().registry(),
            ))
        })
        .as_any()
        .downcast_ref::<WorldState>()
        .unwrap();

    state.find_land(I64Vec2::zero())
}
