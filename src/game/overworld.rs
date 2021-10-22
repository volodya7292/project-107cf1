pub mod block;
pub mod block_component;
pub mod block_model;
pub mod cluster;
pub mod generator;
pub mod structure;
pub mod textured_block_model;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::structure::world::World;
use crate::game::overworld::structure::Structure;
use crate::utils::value_noise::ValueNoise;
use crate::utils::{HashMap, Int, MO_ACQUIRE};
use nalgebra_glm as glm;
use nalgebra_glm::{I64Vec3, Vec3};
use num_derive::FromPrimitive;
use num_traits::FromPrimitive;
use rand::Rng;
use std::sync::atomic::AtomicU8;
use std::sync::{Arc, RwLock};

// TODO Main world - 'The Origin'

const MIN_WORLD_RADIUS: u64 = 2_048;
pub const MAX_WORLD_RADIUS: u64 = 32_000_000;
// const MIN_DIST_BETWEEN_WORLDS: u64 = 400_000_000;
// const MAX_DIST_BETWEEN_WORLDS: u64 = 40_000_000_000;

pub const LOD_LEVELS: usize = 24;

fn sample_world_size(rng: &mut impl rand::Rng) -> u64 {
    const AVG_R: u64 = (MIN_WORLD_RADIUS + MAX_WORLD_RADIUS) / 2;
    const R_HALF_DIST: f64 = ((MAX_WORLD_RADIUS - MIN_WORLD_RADIUS) / 2) as f64;

    let s: f64 = rng.sample(rand_distr::StandardNormal);
    AVG_R + (s / 3.0 * R_HALF_DIST).clamp(-R_HALF_DIST, R_HALF_DIST) as u64
}

#[derive(Copy, Clone, Eq, PartialEq, FromPrimitive)]
#[repr(u8)]
pub enum ClusterState {
    UNLOADED = 0,
    LOADING = 1,
    LOADED = 2,
}

pub struct OverworldCluster {
    pub cluster: RwLock<Cluster>,
    pub state: AtomicU8,
}

impl OverworldCluster {
    pub fn state(&self) -> ClusterState {
        FromPrimitive::from_u8(self.state.load(MO_ACQUIRE)).unwrap()
    }
}

pub struct Overworld {
    seed: u64,
    main_registry: Arc<MainRegistry>,
    value_noise: ValueNoise<u64>,
    pub loaded_clusters: HashMap<I64Vec3, Arc<OverworldCluster>>,
}

impl Overworld {
    pub fn new(registry: &Arc<MainRegistry>, seed: u64) -> Overworld {
        Overworld {
            seed,
            main_registry: Arc::clone(registry),
            value_noise: ValueNoise::new(seed),
            loaded_clusters: Default::default(),
        }
    }

    fn get_world(&self, center_pos: I64Vec3) -> World {
        let center_pos = center_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));

        World::new(
            self.value_noise
                .state()
                .next(center_pos.x)
                .next(center_pos.y)
                .next(center_pos.z)
                .0,
        )
    }

    fn gen_spawn_point(&self) -> I64Vec3 {
        let reg = self.main_registry.registry();
        let st = reg.get_structure(self.main_registry.structure_world()).unwrap();

        let p = self.gen_structure_pos(st, I64Vec3::from_element(0)).0;
        let _world = self.get_world(p);

        todo!()
    }

    /// Returns position of the structure center (within gen-octant corresponding to the cluster position)
    /// and a `bool` indicating whether the structure is actually present there.  
    /// Gen-octant size = `structure.avg_spacing * cluster_size(structure.cluster_level)`.
    ///
    /// `cluster_pos` is a cluster position of level `self.cluster_level`.
    pub fn gen_structure_pos(&self, structure: &Structure, cluster_pos: I64Vec3) -> (I64Vec3, bool) {
        let structure_fit_size = cluster::size(structure.cluster_level());
        let octant_pos = cluster_pos.map(|v| v.div_floor(structure.avg_spacing() as i64));
        let octant_pos_u64 = octant_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));
        let octant_size = structure.avg_spacing() * structure_fit_size;

        let mut rng = self
            .value_noise
            .state()
            .next(structure.cluster_level() as u64)
            .next(octant_pos_u64.x)
            .next(octant_pos_u64.y)
            .next(octant_pos_u64.z)
            .rng();
        let mut present = rng.gen::<bool>();

        let r = (structure_fit_size / 2)..(octant_size - structure_fit_size / 2);
        let dx = rng.gen_range(r.clone());
        let dy = rng.gen_range(r.clone());
        let dz = rng.gen_range(r.clone());
        let center_pos = octant_pos * (octant_size as i64) + I64Vec3::new(dx as i64, dy as i64, dz as i64);

        if present {
            present = structure.check_gen_pos(self, center_pos);
        }

        (center_pos, present)
    }

    /// Find the nearest structure position to the `starting_cluster_pos` position.
    ///
    /// `start_cluster_pos` is a starting cluster position of level `structure.cluster_level`.  
    /// `max_search_radius` is radius in gen-octants of search domain.
    pub fn find_structure_pos(
        &self,
        structure: &Structure,
        start_cluster_pos: I64Vec3,
        max_search_radius: u32,
    ) -> Option<I64Vec3> {
        use std::f32::consts::PI;

        let octant_size = (structure.avg_spacing() * cluster::size(structure.cluster_level())) as i64;
        let phi_units = max_search_radius * 4;
        let theta_units = phi_units * 2;

        for r_i in 0..max_search_radius {
            let r = r_i as f32;

            for phi_i in 0..phi_units {
                let phi = phi_i as f32 / phi_units as f32 * PI;

                for theta_i in 0..theta_units {
                    let theta = theta_i as f32 / theta_units as f32 * 2.0 * PI;

                    let x = r * phi.cos() * theta.sin();
                    let y = r * phi.sin() * theta.cos();
                    let z = r * theta.cos();

                    let v: I64Vec3 = glm::try_convert(Vec3::new(x, y, z)).unwrap();
                    let p = start_cluster_pos + v * octant_size;

                    let result = self.gen_structure_pos(structure, p);
                    if result.1 {
                        return Some(result.0);
                    }
                }
            }
        }

        None
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
