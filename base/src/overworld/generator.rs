use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use bit_vec::BitVec;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, U32Vec3};
use noise::Seedable;
use once_cell::sync::OnceCell;
use parking_lot::RwLock;
use rand::Rng;
use rand_distr::num_traits::Zero;

use crate::main_registry::MainRegistry;
use crate::overworld::position::{BlockPos, ClusterPos};
use crate::overworld::raw_cluster;
use crate::overworld::raw_cluster::RawCluster;
use crate::overworld::structure::world::WorldState;
use crate::overworld::structure::{Structure, StructuresIter};
use crate::utils::noise::HybridNoise;
use crate::utils::white_noise::WhiteNoise;
use crate::utils::{ConcurrentCache, ConcurrentCacheImpl, UInt};

// Note: always set empty blocks to potentially mark the whole cluster as empty

const MAX_STRUCTURE_STATES_SIZE: usize = 1024 * 256; // 256 MB

pub trait StructureCache: Any + Send + Sync + 'static {
    /// Returns the size of the cache in kilobytes
    fn size(&self) -> u32;
    fn as_any(&self) -> &dyn Any;
}

pub struct StructurePos {
    pub octant: I64Vec3,
    pub center_pos: BlockPos,
    pub present: bool,
}

pub struct OverworldGenerator {
    seed: u64,
    white_noise: WhiteNoise,
    main_registry: Arc<MainRegistry>,
    structure_states: ConcurrentCache<(u32, BlockPos), Arc<OnceCell<Box<dyn StructureCache>>>>,
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

    pub fn get_world_seed(&self, center_pos: BlockPos) -> u64 {
        let center_pos = center_pos.0.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));
        self.white_noise
            .state()
            .next(center_pos.x)
            .next(center_pos.y)
            .next(center_pos.z)
            .0
    }

    /// Returns position of the structure center (within gen-octant corresponding to the block position)
    /// and a `bool` indicating whether the structure is actually present there.  
    /// Gen-octant size = `structure.avg_spacing * cluster_size` blocks.
    pub fn gen_structure_pos(&self, structure: &Structure, pos: BlockPos) -> StructurePos {
        let structure_fit_size = UInt::next_multiple_of(structure.max_size().max(), RawCluster::SIZE as u64);
        let octant_size = structure.avg_spacing() * RawCluster::SIZE as u64;
        let octant = pos.0.map(|v| v.div_euclid(octant_size as i64));
        let octant_u64 = octant.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));

        let mut rng = self
            .white_noise
            .state()
            .next(structure.uid() as u64)
            .next(octant_u64.x)
            .next(octant_u64.y)
            .next(octant_u64.z)
            .rng();
        let mut present = rng.gen::<bool>();

        let low_margin = (structure_fit_size / 2).max(structure.min_spacing() / 2);
        let range = low_margin..(octant_size - low_margin);

        let dx = rng.gen_range(range.clone()) as i64;
        let dy = rng.gen_range(range.clone()) as i64;
        let dz = rng.gen_range(range.clone()) as i64;
        let center_pos = octant * (octant_size as i64) + I64Vec3::new(dx, dy, dz);
        let center_pos = BlockPos(center_pos);

        if present {
            present = structure.check_gen_pos(self, center_pos);
        }

        StructurePos {
            octant,
            center_pos,
            present,
        }
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
        let start_octant = start_cluster_pos.map(|v| v.div_euclid(clusters_per_octant));

        let mut queue = VecDeque::with_capacity(volume);
        let traversed_nodes = BitVec::from_elem(volume, false);

        queue.push_back(start_octant);

        StructuresIter {
            generator: self,
            structure,
            start_octant,
            max_search_radius,
            queue,
            traversed_nodes,
        }
    }

    /// Returns center pos of found world
    pub fn gen_world_pos(&self, pos: BlockPos) -> StructurePos {
        let reg = self.main_registry.registry();
        let world_st = reg.get_structure(self.main_registry.structure_world()).unwrap();
        let mut st_pos = self.gen_structure_pos(world_st, pos);

        if st_pos.octant.is_zero() {
            st_pos.present = true;
        }
        st_pos
    }

    pub fn gen_spawn_point(&self) -> BlockPos {
        let reg = self.main_registry.registry();
        let world_st = reg.get_structure(self.main_registry.structure_world()).unwrap();
        let world_pos = self.gen_world_pos(BlockPos(I64Vec3::zeros()));
        let world_seed = self.get_world_seed(world_pos.center_pos);

        let cache = self
            .structure_states
            .get_with((world_st.uid(), world_pos.center_pos), || {
                Arc::new(OnceCell::new())
            });

        let rel_spawn_point = world_st.gen_spawn_point(self, world_seed, cache).unwrap();
        world_pos.center_pos.offset(&rel_spawn_point.0)
    }

    pub fn create_cluster(&self) -> RawCluster {
        RawCluster::new(self.main_registry.registry())
    }

    pub fn generate_cluster(&self, cluster: &mut RawCluster, pos: ClusterPos) {
        let reg = self.main_registry.registry();
        let world_st = reg.get_structure(self.main_registry.structure_world()).unwrap();
        let world_pos = self.gen_world_pos(pos.to_block_pos());

        if !world_pos.present {
            return;
        }

        let world_seed = self.get_world_seed(world_pos.center_pos);
        let cache = self
            .structure_states
            .get_with((world_st.uid(), world_pos.center_pos), || {
                Arc::new(OnceCell::new())
            });

        let rel_cluster_pos = pos
            .to_block_pos()
            .offset(&(-world_pos.center_pos.0))
            .cluster_pos();

        world_st.gen_cluster(self, world_seed, rel_cluster_pos, cluster, cache);
    }
}
