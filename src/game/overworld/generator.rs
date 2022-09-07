use crate::game::main_registry::MainRegistry;
use crate::game::overworld::cluster;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::structure::{Structure, StructuresIter};
use bit_vec::BitVec;
use engine::utils::noise::HybridNoise;
use engine::utils::white_noise::WhiteNoise;
use engine::utils::{ConcurrentCache, ConcurrentCacheImpl, HashMap, UInt};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, U32Vec3};
use noise::Seedable;
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use rand::Rng;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

// Note: always set empty blocks to potentially mark the whole cluster as empty

const MAX_STRUCTURE_STATES_SIZE: usize = 1024 * 256; // 256 MB

pub trait StructureCache: Any + Send + Sync + 'static {
    /// Returns the size of the cache in kilobytes
    fn size(&self) -> u32;
}

pub struct OverworldGenerator {
    seed: u64,
    white_noise: WhiteNoise,
    main_registry: Arc<MainRegistry>,
    structure_states: ConcurrentCache<(u32, I64Vec3), Arc<OnceCell<Box<dyn StructureCache>>>>,
}

impl OverworldGenerator {
    pub fn new(seed: u64, main_registry: &Arc<MainRegistry>) -> Self {
        Self {
            seed,
            white_noise: WhiteNoise::new(seed),
            main_registry: Arc::clone(main_registry),
            structure_states: ConcurrentCache::with_weigher(
                MAX_STRUCTURE_STATES_SIZE,
                |_, v: &Arc<OnceCell<Box<dyn StructureCache>>>| v.get().map(|v| v.size()).unwrap_or(0),
            ),
        }
    }

    pub fn main_registry(&self) -> &Arc<MainRegistry> {
        &self.main_registry
    }

    fn get_world_seed(&self, center_pos: I64Vec3) -> u64 {
        let center_pos = center_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));
        self.white_noise
            .state()
            .next(center_pos.x)
            .next(center_pos.y)
            .next(center_pos.z)
            .0
    }

    /// Returns position of the structure center (within gen-octant corresponding to the cluster position)
    /// and a `bool` indicating whether the structure is actually present there.  
    /// Gen-octant size = `structure.avg_spacing * cluster_size` blocks.
    /// Cluster position is in blocks.
    pub fn gen_structure_pos(&self, structure: &Structure, cluster_pos: I64Vec3) -> (I64Vec3, bool) {
        let structure_fit_size = UInt::next_multiple_of(structure.max_size().max(), cluster::SIZE as u64);
        let octant_size = structure.avg_spacing() * cluster::SIZE as u64;
        let octant_pos = cluster_pos.map(|v| v.div_euclid(octant_size as i64));
        let octant_pos_u64 = octant_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));

        let mut rng = self
            .white_noise
            .state()
            .next(structure.uid() as u64)
            .next(octant_pos_u64.x)
            .next(octant_pos_u64.y)
            .next(octant_pos_u64.z)
            .rng();
        let mut present = rng.gen::<bool>();

        let low_margin = (structure_fit_size / 2).max(structure.min_spacing() / 2);
        let range = low_margin..(octant_size - low_margin);

        let dx = rng.gen_range(range.clone()) as i64;
        let dy = rng.gen_range(range.clone()) as i64;
        let dz = rng.gen_range(range.clone()) as i64;
        let center_pos = octant_pos * (octant_size as i64) + I64Vec3::new(dx, dy, dz);

        if present {
            present = structure.check_gen_pos(self, center_pos);
        }

        (center_pos, present)
    }

    /// Find the nearest structure positions to the `start_cluster_pos` position.  
    /// Returned positions are in blocks.
    ///
    /// `structure` is the structure to search for.
    /// `start_cluster_pos` is the starting cluster position.  
    /// `max_search_radius` is the radius in gen-octants of search domain, must be >= 1.
    /// The gen-octant size is `structure.avg_spacing` clusters.
    pub fn find_nearest_structures<'a>(
        &'a self,
        structure: &'a Structure,
        start_cluster_pos: I64Vec3,
        max_search_radius: u32,
    ) -> StructuresIter<'a> {
        assert!(max_search_radius >= 1);

        let diam = max_search_radius as i64 * 2 - 1;
        let volume = diam.pow(3) as usize;
        let clusters_per_octant = structure.avg_spacing() as i64;
        let start_pos = start_cluster_pos.map(|v| v.div_euclid(clusters_per_octant));

        let mut queue = VecDeque::with_capacity(volume);
        let mut traversed_nodes = BitVec::from_elem(volume, false);

        queue.push_back(start_pos);

        StructuresIter {
            generator: self,
            structure,
            start_cluster_pos,
            max_search_radius,
            queue,
            traversed_nodes,
        }
    }

    pub fn gen_spawn_point(&self) -> I64Vec3 {
        let reg = self.main_registry.registry();
        let st = reg.get_structure(self.main_registry.structure_world()).unwrap();

        let p = self.gen_structure_pos(st, I64Vec3::from_element(0)).0;
        let _world = self.get_world_seed(p);

        todo!()
    }

    pub fn create_cluster(&self) -> Cluster {
        Cluster::new(self.main_registry.registry())
    }

    pub fn add<T: StructureCache>(&self) {
        // StructureCache::
        // T.default();
    }

    pub fn generate_cluster(&self, cluster: &mut Cluster, pos: I64Vec3) {
        let reg = self.main_registry.registry();
        let world = reg.get_structure(self.main_registry.structure_world()).unwrap();
        let (world_center, world_present) = self.gen_structure_pos(world, pos);

        if !world_present {
            return;
        }

        let world_seed = self.get_world_seed(world_center);
        let mut cache = self
            .structure_states
            .get_with((world.uid(), world_center), || Arc::new(OnceCell::new()));

        world.gen_cluster(self, world_seed, world_center, cluster, cache);

        //
        // let noise = HybridNoise::<3, 1, noise::SuperSimplex>::new(noise::SuperSimplex::new().set_seed(0));
        //
        // // NoiseBuilder::cellular2_2d_offset().generate().
        // // let noise = NoiseBuilder::gradient_3d_offset(
        // //     pos.x as f32 / (entry_size as f32),
        // //     cluster::SIZE,
        // //     pos.y as f32 / (entry_size as f32),
        // //     cluster::SIZE,
        // //     pos.z as f32 / (entry_size as f32),
        // //     cluster::SIZE,
        // // )
        // // .with_seed(0)
        // // .with_freq(1.0 / 50.0 * entry_size as f32)
        // // .generate();
        //
        // let block_empty = self.main_registry.block_empty();
        // let block_default = self.main_registry.block_default();
        //
        // let sample = |point: DVec3, freq: f64| -> f64 { noise.sample(point, freq, 0.5) };
        //
        // for x in 0..cluster::SIZE {
        //     for y in 0..cluster::SIZE {
        //         for z in 0..cluster::SIZE {
        //             let xyz = U32Vec3::new(x as u32, y as u32, z as u32);
        //             let posf: DVec3 = glm::convert(glm::convert::<U32Vec3, I64Vec3>(xyz) + pos);
        //
        //             let n = sample(posf, 0.05);
        //             let n = (n + 2.0) * ((posf.y - 30.0) / 128.0);
        //
        //             if n < 0.5 {
        //                 cluster.set(&xyz, block_default).build();
        //             } else {
        //                 cluster.set(&xyz, block_empty).build();
        //             }
        //
        //             // if posf.x.abs().max(posf.z.abs()) < posf.y {
        //             //     cluster.set_block(xyz, block_default).build();
        //             // }
        //
        //             // let n = noise.0
        //         }
        //     }
        // }

        //
        // let sample_noise =
        //     |x, y, z| -> f32 { noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0 };
        //
        // let index = |x: usize, y: usize, z: usize| -> usize {
        //     (x * cluster::SIZE * cluster::SIZE + y * cluster::SIZE + z) as usize
        // };
        //
        // let mut densities = vec![0_f32; cluster::VOLUME];
        // let mat = if pos.x > 0 { 0 } else { 1 };
        //
        // for x in 0..(cluster::SIZE) {
        //     for y in 0..(cluster::SIZE) {
        //         for z in 0..(cluster::SIZE) {
        //             let n_v = sample_noise(x, y, z);
        //
        //             /*
        //
        //             vec2 coord = fragCoord / 2000.0;
        //
        //             float c = f(coord);
        //
        //             c += 0.9;
        //             c *= (fragCoord.y / 450.0);
        //
        //             if (c > 0.9) {
        //                 //c = 1.0;
        //             } else {
        //                 //c = 0.0;
        //             }
        //
        //             fragColor = vec4(c, c, c, 1.0);
        //
        //
        //              */
        //
        //             // let n_v = ((n_v as f32 + (64 - (pos.y + y as i32) * (node_size as i32)) as f32 / 10.0) / 2.0)
        //             //     .max(0.0)
        //             //     .min(1.0);
        //
        //             let p0 = na::Vector3::new(0.0, 0.0, 0.0);
        //             let d = na::Vector3::new(pos.x as f32 + x as f32, y as f32, pos.z as f32 + z as f32);
        //             let h = (((d - p0).magnitude() * node_size as f32 + 32.0) / 64.0).clamp(0.0, 1.0);
        //
        //             // let h = ((y as f32 * node_size as f32 + 32.0) / 65.0).clamp(0.0, 1.0);
        //
        //             let n_v = (n_v + 2.0) * (1.0 - h);
        //
        //             densities[index(x, y, z)] = n_v.clamp(0.0, 1.0);
        //         }
        //     }
        // }
        //
        // normalize_densities(&densities, na::convert(na::Vector3::from_element(cluster::SIZE)));
        //
        // let mut points = Vec::<cluster::DensityPointInfo>::with_capacity(cluster::VOLUME);
        //
        // for x in 0..(cluster::SIZE) {
        //     for y in 0..(cluster::SIZE) {
        //         for z in 0..(cluster::SIZE) {
        //             points.push(cluster::DensityPointInfo {
        //                 pos: [x as u8, y as u8, z as u8, 0],
        //                 point: cluster::DensityPoint {
        //                     density: (densities[index(x, y, z)] * 255.0) as u8,
        //                     material: mat,
        //                 },
        //             });
        //         }
        //     }
        // }
        //
        // points
    }
}
