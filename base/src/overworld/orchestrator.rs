use crate::execution::default_queue;
use crate::execution::virtual_processor::{VirtualProcessor, VirtualTask};
use crate::overworld::cluster_part_set::ClusterPartSet;
use crate::overworld::generator::OverworldGenerator;
use crate::overworld::position::{BlockPos, ClusterPos};
use crate::overworld::raw_cluster::RawCluster;
use crate::overworld::{
    ClusterState, ClusterStateEnum, LoadedClusters, Overworld, OverworldCluster, TrackingCluster,
};
use common::glm;
use common::glm::DVec2;
use common::parking_lot::RwLockWriteGuard;
use common::types::{HashMap, HashSet};
use common::{MO_RELAXED, MO_RELEASE};
use glm::DVec3;
use smallvec::SmallVec;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Manages overworld: adds/removes new clusters, loads them in parallel, etc.
pub struct OverworldOrchestrator {
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    loaded_clusters: LoadedClusters,
    overworld_generator: Arc<OverworldGenerator>,
    r_clusters: HashMap<ClusterPos, RCluster>,
    cluster_load_processor: VirtualProcessor,
    cluster_compression_processor: VirtualProcessor,
}

struct RCluster {
    load_task: Option<VirtualTask<()>>,
    compress_task: Option<VirtualTask<()>>,
}

struct ClusterPosDistance {
    pos: ClusterPos,
    distance: f64,
}

fn calc_cluster_layout(
    stream_pos: &DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
) -> HashSet<ClusterPos> {
    let cr = (xz_render_distance / RawCluster::SIZE as u64) as i64 + 1;
    let cl_size = RawCluster::SIZE as i64;
    let c_stream_pos = BlockPos::from_f64(stream_pos).cluster_pos();

    let mut layout = HashSet::with_capacity((cr * 2 + 1).pow(3) as usize);

    for x in -cr..(cr + 1) {
        for y in -cr..(cr + 1) {
            for z in -cr..(cr + 1) {
                let cp = c_stream_pos.offset(&glm::vec3(x, y, z));
                let center = cp.get().add_scalar(cl_size / 2);

                let d = stream_pos - glm::convert::<_, DVec3>(center);

                if (d.x / xz_render_distance as f64).powi(2)
                    + (d.y / y_render_distance as f64).powi(2)
                    + (d.z / xz_render_distance as f64).powi(2)
                    <= 1.0
                {
                    layout.insert(cp);
                }
            }
        }
    }

    layout
}

pub fn get_side_clusters(pos: &ClusterPos) -> SmallVec<[ClusterPos; 26]> {
    let mut neighbours = SmallVec::<[ClusterPos; 26]>::new();

    for x in -1..2 {
        for y in -1..2 {
            for z in -1..2 {
                if x == 0 && y == 0 && z == 0 {
                    continue;
                }
                let pos2 = pos.offset(&glm::vec3(x, y, z));
                neighbours.push(pos2);
            }
        }
    }

    neighbours
}

struct ClustersDiff {
    sorted_layout: Vec<ClusterPosDistance>,
    to_add: Vec<ClusterPos>,
    to_remove: Vec<ClusterPos>,
}

fn get_clusters_difference(
    stream_pos: &DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    r_clusters: &HashMap<ClusterPos, RCluster>,
) -> ClustersDiff {
    let xz_ext_distance = xz_render_distance + RawCluster::SIZE as u64 * 2;
    let y_ext_distance = y_render_distance + RawCluster::SIZE as u64 * 2;
    let layout = calc_cluster_layout(stream_pos, xz_render_distance, y_render_distance);

    let mut sorted_layout: Vec<_> = layout
        .iter()
        .map(|p| ClusterPosDistance {
            pos: *p,
            // distance: glm::distance2(
            //     &glm::convert::<_, DVec3>(p.get().add_scalar(RawCluster::SIZE as i64 / 2)),
            //     &stream_pos,
            // ),
            distance: glm::distance2(
                &glm::convert::<_, DVec2>(p.get().xy().add_scalar(RawCluster::SIZE as i64 / 2)),
                &stream_pos.xy(),
            ),
        })
        .collect();
    // Sort by lowest distance to streaming position
    sorted_layout.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));

    let mut res = ClustersDiff {
        sorted_layout,
        to_add: Vec::with_capacity(8192),
        to_remove: Vec::with_capacity(8192),
    };

    // Collect new clusters
    for d_pos in &res.sorted_layout {
        if !r_clusters.contains_key(&d_pos.pos) {
            res.to_add.push(d_pos.pos);
        }
    }

    // Collect clusters to remove
    for pos in r_clusters.keys() {
        let in_layout = layout.contains(pos);

        let center = pos.get().add_scalar(RawCluster::SIZE as i64 / 2);
        let d = stream_pos - glm::convert::<_, DVec3>(center);

        // Do not remove clusters that are still in extended render distance to avoid sudden loads/unloads
        if !in_layout
            && (d.x / xz_ext_distance as f64).powi(2)
                + (d.y / y_ext_distance as f64).powi(2)
                + (d.z / xz_ext_distance as f64).powi(2)
                > 1.0
        {
            res.to_remove.push(*pos);
        }
    }

    res
}

fn add_new_clusters(
    positions: &[ClusterPos],
    r_clusters: &mut HashMap<ClusterPos, RCluster>,
    o_clusters: &mut HashMap<ClusterPos, OverworldCluster>,
) {
    for pos in positions {
        r_clusters.insert(
            *pos,
            RCluster {
                load_task: None,
                compress_task: None,
            },
        );
        o_clusters.insert(*pos, OverworldCluster::new());
    }
}

fn remove_clusters(
    to_remove: &[ClusterPos],
    r_clusters: &mut HashMap<ClusterPos, RCluster>,
    o_clusters: &mut HashMap<ClusterPos, OverworldCluster>,
) {
    for pos in to_remove {
        r_clusters.remove(pos);
        o_clusters.remove(pos);
    }
}

pub struct OverworldUpdateResult {
    /// When a cluster C has dirty parts, respective auxiliary parts of neighbour clusters
    /// must be updated with cells from opposite parts of C.  
    /// This represents clusters and their respective dirty parts
    pub dirty_clusters_parts: HashMap<ClusterPos, ClusterPartSet>,
    pub new_clusters: Vec<ClusterPos>,
    pub removed_clusters: Vec<ClusterPos>,
}

impl OverworldOrchestrator {
    const MIN_XZ_RENDER_DISTANCE: u64 = 128;
    const MAX_XZ_RENDER_DISTANCE: u64 = 1024;
    const MIN_Y_RENDER_DISTANCE: u64 = 128;
    const MAX_Y_RENDER_DISTANCE: u64 = 1024;
    const MIN_UNCOMPRESSED_DISTANCE: u64 = 128;
    const IDLE_TIME_FOR_COMPRESSION: Duration = Duration::from_secs(4);

    pub fn new(overworld: &Overworld) -> Self {
        Self {
            stream_pos: Default::default(),
            xz_render_distance: 128,
            y_render_distance: 128,
            loaded_clusters: Arc::clone(overworld.loaded_clusters()),
            overworld_generator: Arc::clone(overworld.generator()),
            r_clusters: Default::default(),
            cluster_load_processor: VirtualProcessor::new(&default_queue().unwrap()),
            cluster_compression_processor: VirtualProcessor::new(&default_queue().unwrap()),
        }
    }

    pub fn set_xz_render_distance(&mut self, dist: u64) {
        self.xz_render_distance = dist.clamp(Self::MIN_XZ_RENDER_DISTANCE, Self::MAX_XZ_RENDER_DISTANCE);
    }

    pub fn set_y_render_distance(&mut self, dist: u64) {
        self.y_render_distance = dist.clamp(Self::MIN_Y_RENDER_DISTANCE, Self::MAX_Y_RENDER_DISTANCE);
    }

    pub fn set_stream_pos(&mut self, pos: DVec3) {
        self.stream_pos = pos;
    }

    pub fn loaded_clusters(&self) -> &LoadedClusters {
        &self.loaded_clusters
    }

    pub fn update(&mut self) -> OverworldUpdateResult {
        // Lock loaded_clusters so no accessor can modify clusters
        let mut o_clusters = self.loaded_clusters.write();
        let r_clusters = &mut self.r_clusters;

        let clusters_diff = get_clusters_difference(
            &self.stream_pos,
            self.xz_render_distance,
            self.y_render_distance,
            &r_clusters,
        );

        // Cancel tasks associated with removed clusters
        for d_pos in &clusters_diff.to_remove {
            let cluster = r_clusters.get_mut(d_pos).unwrap();
            if let Some(task) = cluster.load_task.take() {
                task.cancel();
            }
        }

        // 1. Apply clusters difference
        remove_clusters(&clusters_diff.to_remove, r_clusters, &mut o_clusters);
        add_new_clusters(&clusters_diff.to_add, r_clusters, &mut o_clusters);

        // Write access is not needed anymore
        let o_clusters = RwLockWriteGuard::downgrade(o_clusters);

        // 2. Load new clusters
        // ------------------------------------------------------------------------------------------------------
        for d_pos in clusters_diff.sorted_layout.iter() {
            let pos = d_pos.pos;
            let o_cluster = o_clusters.get(&pos).unwrap();
            let r_cluster = r_clusters.get_mut(&pos).unwrap();
            let state = o_cluster.state();

            if state == ClusterStateEnum::Loaded {
                continue;
            }
            if r_cluster.load_task.is_some() {
                continue;
            }

            let t_cluster = Arc::clone(&o_cluster.cluster);
            let generator = Arc::clone(&self.overworld_generator);

            let load_task = self.cluster_load_processor.spawn(move || {
                let mut cluster = RawCluster::new();
                generator.generate_cluster(&mut cluster, pos);
                let mut t_cluster = t_cluster.write();
                *t_cluster = ClusterState::Ready(TrackingCluster::new(cluster, ClusterPartSet::ALL));
            });
            r_cluster.load_task = Some(load_task);
        }

        // 3. Check for previous finished load tasks
        for (pos, rcl) in &mut *r_clusters {
            if rcl.load_task.as_mut().map_or(true, |v| !v.is_finished()) {
                continue;
            };
            rcl.load_task = None;

            let o_cluster = o_clusters.get(pos).unwrap();
            o_cluster.state.store(ClusterStateEnum::Loaded as u32, MO_RELEASE);
            o_cluster.dirty.store(true, MO_RELEASE);
        }

        // 4. Collect dirty clusters
        // ------------------------------------------------------------------------------------------------------
        let mut dirty_clusters = HashMap::with_capacity(1024);

        for (pos, o_cluster) in &*o_clusters {
            if !o_cluster.state().is_loaded() {
                continue;
            }
            if !o_cluster.dirty.load(MO_RELAXED) {
                continue;
            }
            o_cluster.dirty.store(false, MO_RELAXED);

            let mut cluster = o_cluster.cluster.write();
            if !cluster.is_loaded() {
                continue;
            }

            let dirty_parts = cluster.take_dirty_parts();
            dirty_clusters.insert(*pos, dirty_parts);
        }

        // 5. Check dirty clusters for active blocks
        // ------------------------------------------------------------------------------------------------------
        for (pos, _) in &dirty_clusters {
            let o_cluster = o_clusters.get(pos).unwrap();
            let t_cluster = o_cluster.cluster.read();
            let has_active_blocks = t_cluster.has_active_blocks();
            o_cluster.has_active_blocks.store(has_active_blocks, MO_RELAXED);
        }

        // 6. Compress redundant clusters
        // ------------------------------------------------------------------------------------------------------
        // let n_max_uncompressed_clusters = (self.xz_render_distance.pow(2) as f64 * 3.14) as usize;
        // let uncompressed_count = 0;
        let curr_time = Instant::now();

        for (pos, o_cluster) in &*o_clusters {
            let dist = glm::distance2(&glm::convert::<_, DVec3>(*pos.get()), &self.stream_pos);
            if dist < Self::MIN_UNCOMPRESSED_DISTANCE.pow(2) as f64 {
                // Nearby clusters are always uncompressed for minimum access latency
                continue;
            }

            {
                let t_cluster = o_cluster.cluster.read();

                if let ClusterState::Ready(t_cluster) = &*t_cluster {
                    if curr_time.duration_since(t_cluster.last_used_time()) <= Self::IDLE_TIME_FOR_COMPRESSION
                    {
                        continue;
                    }
                }

                if t_cluster.is_initial() || t_cluster.is_compressed() {
                    // The cluster is not loaded or is already compressed
                    continue;
                }
            }

            let r_cluster = r_clusters.get_mut(&pos).unwrap();
            if r_cluster
                .compress_task
                .as_ref()
                .map_or(false, |v| !v.is_finished())
            {
                // The previous compression task is not finished
                continue;
            }

            let task = {
                let t_cluster = Arc::clone(&o_cluster.cluster);
                self.cluster_compression_processor.spawn(move || {
                    let mut t_cluster = t_cluster.write();
                    t_cluster.compress();
                })
            };
            r_cluster.compress_task = Some(task);
        }

        OverworldUpdateResult {
            dirty_clusters_parts: dirty_clusters,
            new_clusters: clusters_diff.to_add,
            removed_clusters: clusters_diff.to_remove,
        }
    }
}
