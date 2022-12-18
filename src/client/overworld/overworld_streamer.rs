use std::collections::hash_map;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};
use std::sync::Arc;
use std::time::Instant;

use entity_data::{EntityId, SystemHandler};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, U8Vec3, Vec3};
use parking_lot::{Mutex, RwLock, RwLockUpgradableReadGuard, RwLockWriteGuard};
use smallvec::SmallVec;

use core::main_registry::MainRegistry;
use core::overworld::cluster_dirty_parts::ClusterDirtySides;
use core::overworld::facing::Facing;
use core::overworld::generator::OverworldGenerator;
use core::overworld::liquid_state::LiquidState;
use core::overworld::occluder::Occluder;
use core::overworld::position::{BlockPos, ClusterPos};
use core::overworld::raw_cluster::{CellInfo, RawCluster};
use core::overworld::{
    generator, raw_cluster, Cluster, ClusterState, LoadedClusters, Overworld, OverworldCluster,
};
use engine::ecs::component;
use engine::queue::intensive_queue;
use engine::renderer::{Renderer, VertexMeshObject};
use engine::unwrap_option;
use engine::utils::{HashMap, HashSet, MO_ACQUIRE, MO_RELAXED, MO_RELEASE};
use vk_wrapper as vkw;

use crate::client::overworld::raw_cluster_ext::{ClientRawCluster, ClusterMeshes};
use crate::resource_mapping::ResourceMapping;

pub const FORCED_LOAD_RANGE: usize = 128;

pub struct OverworldStreamer {
    device: Arc<vkw::Device>,
    cluster_mat_pipeline: u32,
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    loaded_clusters: LoadedClusters,
    res_map: Arc<ResourceMapping>,
    overworld_generator: Arc<OverworldGenerator>,
    rclusters: HashMap<ClusterPos, RCluster>,
    clusters_entities_to_remove: Vec<EntityId>,
    clusters_entities_to_add: Vec<ClusterPos>,
    curr_loading_clusters_n: Arc<AtomicU32>,
    curr_clusters_intrinsics_updating_n: Arc<AtomicU32>,
    curr_clusters_meshes_updating_n: Arc<AtomicU32>,
}

const CLUSTER_ALL_NEIGHBOURS_3X3: u32 = ((1 << 27) - 1) & !(1 << 13);

#[derive(Default)]
struct RClusterEntities {
    solid: EntityId,
    translucent: EntityId,
}

impl RClusterEntities {
    fn is_null(&self) -> bool {
        self.solid == EntityId::NULL && self.translucent == EntityId::NULL
    }
}

struct RCluster {
    creation_time: Instant,
    entities: RClusterEntities,
    /// Intrinsics mask of clusters whose intrinsics data is currently being pasted into this cluster
    updating_outer_intrinsics: Arc<AtomicU32>,
    /// Whether this cluster is invisible due to full intrinsics by the neighbour clusters
    occluded: AtomicBool,
    empty: Arc<AtomicBool>,
    /// A mask of 3x3 'sides' of outer intrinsics which are needed to be cleared
    needs_intrinsics_clear_at: u32,
    /// A mask of 3x3 'sides' which are needed to be filled with intrinsics of neighbour clusters
    needs_intrinsics_fill_at: u32,
    /// Whether a particular Facing of this cluster is fully occluded by another cluster
    edge_intrinsics: Arc<AtomicU8>,
    /// Whether `edge_intrinsics` has been determined, only then `occluded` can be set
    edge_intrinsics_determined: Arc<AtomicBool>,
    /// All conditions are met for the vertex mesh to be updated
    mesh_can_be_updated: Arc<AtomicBool>,
    /// Mesh has been updated and it needs to be updated inside the VertexMesh scene component
    mesh_changed: Arc<AtomicBool>,

    meshes: Arc<Mutex<ClusterMeshes>>,
}

struct ClusterPosDistance {
    pos: ClusterPos,
    distance: f64,
}

fn calc_cluster_layout(
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
) -> HashSet<ClusterPos> {
    let cr = (xz_render_distance / RawCluster::SIZE as u64 / 2) as i64;
    let cl_size = RawCluster::SIZE as i64;
    let c_stream_pos = BlockPos::from_f64(&stream_pos).cluster_pos();

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

fn get_side_clusters(pos: &ClusterPos) -> SmallVec<[ClusterPos; 26]> {
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

fn get_side_cluster_by_facing(pos: ClusterPos, facing: Facing) -> ClusterPos {
    pos.offset(&glm::convert(facing.direction()))
}

fn neighbour_dir_index(pos: &ClusterPos, target: &ClusterPos) -> usize {
    let diff = ((target.get() - pos.get()) / RawCluster::SIZE as i64).add_scalar(1);
    diff.x as usize * 9 + diff.y as usize * 3 + diff.z as usize
}

fn look_cube_directions(dir: DVec3) -> [I32Vec3; 3] {
    [
        I32Vec3::new(-dir.x.signum() as i32, 0, 0),
        I32Vec3::new(0, -dir.y.signum() as i32, 0),
        I32Vec3::new(0, 0, -dir.z.signum() as i32),
    ]
}

fn is_cluster_in_forced_load_range(pos: &ClusterPos, stream_pos: &DVec3) -> bool {
    let center_pos: DVec3 = glm::convert(pos.get().add_scalar(RawCluster::SIZE as i64 / 2));
    let distance = glm::distance(&center_pos, &stream_pos);

    distance <= FORCED_LOAD_RANGE as f64
}

/// Checks if cluster at `pos` is occluded at all sides by the neighbour clusters
fn is_cluster_visibly_occluded(
    rclusters: &HashMap<ClusterPos, RCluster>,
    pos: ClusterPos,
    stream_pos: DVec3,
) -> bool {
    let center_pos: DVec3 = glm::convert(pos.get().add_scalar(RawCluster::SIZE as i64 / 2));
    let dir_to_cluster_unnorm = center_pos - stream_pos;

    if dir_to_cluster_unnorm.magnitude() <= FORCED_LOAD_RANGE as f64 {
        return false;
    }

    let mut edges_occluded = true;

    for i in 0..6 {
        let facing = Facing::from_u8(i);
        let p = get_side_cluster_by_facing(pos, facing);

        if let Some(rcl) = rclusters.get(&p) {
            let edge_occluded = ((rcl.edge_intrinsics.load(MO_RELAXED) >> (facing.mirror() as u8)) & 1) == 1;
            edges_occluded &= edge_occluded;
        } else {
            edges_occluded = false;
            break;
        }

        if !edges_occluded {
            break;
        }
    }

    edges_occluded
}

impl OverworldStreamer {
    const MIN_XZ_RENDER_DISTANCE: u64 = 128;
    const MAX_XZ_RENDER_DISTANCE: u64 = 1024;
    const MIN_Y_RENDER_DISTANCE: u64 = 128;
    const MAX_Y_RENDER_DISTANCE: u64 = 512;

    pub fn new(
        re: &Renderer,
        cluster_mat_pipeline: u32,
        overworld: &Overworld,
        res_map: Arc<ResourceMapping>,
    ) -> Self {
        Self {
            device: Arc::clone(re.device()),
            cluster_mat_pipeline,
            stream_pos: Default::default(),
            xz_render_distance: 128,
            y_render_distance: 128,
            loaded_clusters: Arc::clone(overworld.loaded_clusters()),
            res_map,
            overworld_generator: Arc::clone(overworld.generator()),
            rclusters: Default::default(),
            clusters_entities_to_remove: Default::default(),
            clusters_entities_to_add: Default::default(),
            curr_loading_clusters_n: Default::default(),
            curr_clusters_intrinsics_updating_n: Default::default(),
            curr_clusters_meshes_updating_n: Default::default(),
        }
    }

    pub fn set_xz_render_distance(&mut self, dist: u64) {
        self.xz_render_distance = dist
            .min(Self::MAX_XZ_RENDER_DISTANCE)
            .max(Self::MIN_XZ_RENDER_DISTANCE);
    }

    pub fn set_y_render_distance(&mut self, dist: u64) {
        self.y_render_distance = dist
            .min(Self::MAX_Y_RENDER_DISTANCE)
            .max(Self::MIN_Y_RENDER_DISTANCE);
    }

    pub fn set_stream_pos(&mut self, pos: DVec3) {
        self.stream_pos = pos;
    }

    pub fn loaded_clusters(&self) -> &LoadedClusters {
        &self.loaded_clusters
    }

    /// Generates new clusters and their content. Updates and optimizes overworld cluster layout.
    /// Returns positions of clusters which have been changed by `Overworld` or generated.
    pub fn update(&mut self) -> HashSet<ClusterPos> {
        // Note: the work this method schedules on other threads is supposed to run for 20 ms in total.
        let layout = calc_cluster_layout(self.stream_pos, self.xz_render_distance, self.y_render_distance);
        let rclusters = &mut self.rclusters;
        let mut oclusters = self.loaded_clusters.upgradable_read();
        let curr_t = Instant::now();

        let mut sorted_layout: Vec<_> = layout
            .iter()
            .map(|p| ClusterPosDistance {
                pos: *p,
                distance: glm::distance2(
                    &glm::convert::<_, DVec3>(p.get().add_scalar(RawCluster::SIZE as i64 / 2)),
                    &self.stream_pos,
                ),
            })
            .collect();
        sorted_layout.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));

        let num_threads = intensive_queue().current_num_threads() as u32;

        // Cluster load time ~ 2ms. Each update load clusters for ~ 10ms. 10 / 2 ~= 5 clusters per thread.
        let max_loading_clusters_in_progress = num_threads * 5;
        // Cluster mesh update time ~ 2ms. Each update load updates meshes for another 10ms. 10 / 2 ~= 5 clusters per thread.
        let max_updating_clusters_in_progress = num_threads * 5;

        // Add new clusters according to layout & Remove unnecessary clusters
        {
            let mut removed_rclusters = Vec::with_capacity(rclusters.len());

            rclusters.retain(|p, rcl| {
                let in_layout = layout.contains(p);
                let timeout = (curr_t - rcl.creation_time).as_secs() >= 3;
                let do_preserve = in_layout || !timeout;

                if !do_preserve && !rcl.entities.is_null() {
                    self.clusters_entities_to_remove
                        .extend([rcl.entities.solid, rcl.entities.translucent]);
                    removed_rclusters.push(*p);
                }

                do_preserve
            });

            let mut oclusters_to_remove = Vec::with_capacity(oclusters.len());

            for (p, ocluster) in &*oclusters {
                let in_layout = rclusters.contains_key(p);

                if in_layout {
                    let rcl = rclusters.get_mut(p).unwrap();
                    // Do not remove nearby empty/occluded clusters up to FORCED_LOAD_RANGE
                    // to allow fast access to blocks near the streaming position.
                    let is_empty = rcl.empty.load(MO_RELAXED);
                    let is_occluded = rcl.occluded.load(MO_RELAXED);

                    let visible =
                        (!is_empty && !is_occluded) || is_cluster_in_forced_load_range(p, &self.stream_pos);

                    if !visible {
                        if !rcl.entities.is_null() {
                            self.clusters_entities_to_remove
                                .extend([rcl.entities.solid, rcl.entities.translucent]);
                            rcl.entities = Default::default();
                        }

                        // Note: it is valid to transition to this state because the cluster is guaranteed to be loaded
                        // due to its visibility because RCluster in its initial state is not 'empty'.

                        let new_state = if is_empty {
                            ClusterState::OffloadedEmpty
                        } else {
                            ClusterState::OffloadedOccluded
                        };
                        ocluster.state.store(new_state as u32, MO_RELAXED);

                        *ocluster.cluster.write() = None;
                    }
                } else {
                    oclusters_to_remove.push(*p);
                    ocluster.state.store(ClusterState::Discarded as u32, MO_RELAXED);
                }
            }

            // Remove clusters separately to prevent possible deadlocks
            let mut oclusters_write = RwLockUpgradableReadGuard::upgrade(oclusters);
            for pos in oclusters_to_remove {
                oclusters_write.remove(&pos);
            }

            for p in &layout {
                if let hash_map::Entry::Vacant(e) = rclusters.entry(*p) {
                    e.insert(RCluster {
                        creation_time: curr_t,
                        entities: Default::default(),
                        updating_outer_intrinsics: Default::default(),
                        occluded: Default::default(),
                        empty: Default::default(),
                        needs_intrinsics_clear_at: 0,
                        needs_intrinsics_fill_at: CLUSTER_ALL_NEIGHBOURS_3X3,
                        edge_intrinsics: Default::default(),
                        edge_intrinsics_determined: Default::default(),
                        mesh_can_be_updated: Default::default(),
                        mesh_changed: Default::default(),
                        meshes: Arc::new(Default::default()),
                    });
                    self.clusters_entities_to_add.push(*p);
                }
                if let hash_map::Entry::Vacant(e) = oclusters_write.entry(*p) {
                    e.insert(Arc::new(OverworldCluster::new()));
                }
            }
            oclusters = RwLockWriteGuard::downgrade_to_upgradable(oclusters_write);

            // Mask neighbour rclusters of removed rclusters for clearing their outer intrinsics
            for removed_pos in removed_rclusters {
                for p in get_side_clusters(&removed_pos) {
                    if let Some(rcluster) = rclusters.get_mut(&p) {
                        rcluster.needs_intrinsics_clear_at |= 1 << neighbour_dir_index(&p, &removed_pos);
                    }
                }
            }
        }

        // Load clusters
        {
            for p in &sorted_layout {
                if self.curr_loading_clusters_n.load(MO_ACQUIRE) >= max_loading_clusters_in_progress {
                    break;
                }
                if rclusters[&p.pos].empty.load(MO_RELAXED) || rclusters[&p.pos].occluded.load(MO_RELAXED) {
                    continue;
                }

                // TODO FIXME: there is inconsistency between `rclusters[].empty` and `ocluster.state` when the cluster is offloaded

                let ocluster = &oclusters[&p.pos];
                let state = ocluster.state();
                if matches!(state, ClusterState::Loaded | ClusterState::Loading) {
                    // if ocluster.state() != ClusterState::Initial {
                    continue;
                }

                let curr_loading_clusters_n = Arc::clone(&self.curr_loading_clusters_n);
                let generator = Arc::clone(&self.overworld_generator);
                let ocluster = Arc::clone(&ocluster);
                let pos = p.pos;

                ocluster.state.store(ClusterState::Loading as u32, MO_RELAXED);
                curr_loading_clusters_n.fetch_add(1, MO_ACQUIRE);

                intensive_queue().spawn(move || {
                    if ocluster.state() == ClusterState::Discarded {
                        curr_loading_clusters_n.fetch_sub(1, MO_RELAXED);
                        return;
                    }

                    let mut cluster = generator.create_cluster();

                    generator.generate_cluster(&mut cluster, pos);

                    *ocluster.cluster.write() = Some(Cluster::new(cluster));

                    // Note: use CAS to account for DISCARDED state
                    let _ = ocluster.state.compare_exchange(
                        ClusterState::Loading as u32,
                        ClusterState::Loaded as u32,
                        MO_RELEASE,
                        MO_RELAXED,
                    );
                    ocluster.dirty.store(true, MO_RELEASE);

                    curr_loading_clusters_n.fetch_sub(1, MO_RELEASE);
                });
            }
        }

        let preloaded_clusters_states: HashMap<_, _> =
            oclusters.iter().map(|(p, v)| (*p, v.state())).collect();

        // Generate meshes
        // -------------------------------------------------------------------------------------
        let mut dirty_clusters_positions = HashSet::with_capacity(oclusters.len());

        // Mark intrinsics changes
        for (pos, ocluster) in oclusters.iter() {
            if preloaded_clusters_states[pos] != ClusterState::Loaded || !ocluster.dirty.load(MO_RELAXED) {
                continue;
            }
            ocluster.dirty.store(false, MO_RELAXED);
            dirty_clusters_positions.insert(*pos);

            let rcluster = rclusters.get_mut(pos).unwrap();
            let dirty_parts = {
                let mut cluster = ocluster.cluster.write();
                mem::replace(
                    &mut cluster.as_mut().unwrap().dirty_parts,
                    ClusterDirtySides::none(),
                )
            };

            // Set a flag for updating the mesh when possible (eg. when outer intrinsics is filled)
            rcluster.mesh_can_be_updated.store(true, MO_RELAXED);

            // Set the need to update intrinsics of sides of neighbour clusters
            for i in dirty_parts.iter_sides() {
                let changed_dir: I64Vec3 = glm::convert(raw_cluster::neighbour_index_to_dir(i));

                let p0 = pos.get() / (RawCluster::SIZE as i64);
                let p1 = p0 + changed_dir;

                let min = p0.inf(&p1);
                let max = p0.sup(&p1);

                for x in min.x..=max.x {
                    for y in min.y..=max.y {
                        for z in min.z..=max.z {
                            let p = ClusterPos::new(glm::vec3(x, y, z) * (RawCluster::SIZE as i64));
                            if p == *pos {
                                continue;
                            }
                            if let Some(rcl) = rclusters.get_mut(&p) {
                                rcl.needs_intrinsics_fill_at |= 1 << neighbour_dir_index(&p, pos);
                            }
                        }
                    }
                }
            }
        }

        // Determine occlusion of each cluster & whether respective meshes can be updated
        for pos in oclusters.keys() {
            let rcluster = &rclusters[pos];

            if rcluster.edge_intrinsics_determined.load(MO_RELAXED) {
                rcluster.occluded.store(
                    is_cluster_visibly_occluded(rclusters, *pos, self.stream_pos),
                    MO_RELAXED,
                );
            }

            let rcluster = rclusters.get_mut(pos).unwrap();

            if rcluster.needs_intrinsics_fill_at != 0 || rcluster.needs_intrinsics_clear_at != 0 {
                // Do not update mesh if outer intrinsics is not filled yet
                rcluster.mesh_can_be_updated.store(false, MO_RELAXED);
            }
        }

        // Sort clusters by distance to streaming position
        let mut keys: Vec<_> = oclusters
            .keys()
            .map(|v| {
                (
                    *v,
                    (glm::convert::<_, DVec3>(*v.get()) - self.stream_pos).magnitude_squared(),
                )
            })
            .collect();
        keys.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));

        // Schedule tasks of updating outer intrinsics of respective clusters
        for (pos, _) in &keys {
            if self.curr_clusters_intrinsics_updating_n.load(MO_RELAXED) >= max_updating_clusters_in_progress
            {
                break;
            }

            let rcluster = unwrap_option!(rclusters.get(pos), continue);
            let mut needs_intrinsics_fill_at = rcluster.needs_intrinsics_fill_at;
            let needs_intrinsics_clear_at = rcluster.needs_intrinsics_clear_at;

            if needs_intrinsics_fill_at == 0 && needs_intrinsics_clear_at == 0 {
                continue;
            }

            let ocluster = unwrap_option!(oclusters.get(pos), continue);

            if ocluster.state() != ClusterState::Loaded {
                continue;
            }
            let cluster = Arc::clone(&ocluster.cluster);

            let mut ready_to_fill = true;
            let mut fill_neighbours = SmallVec::<[_; 26]>::new();

            for p in get_side_clusters(&pos) {
                let mask = 1 << neighbour_dir_index(pos, &p);

                if needs_intrinsics_fill_at & mask == 0 {
                    continue;
                }
                if let Some(rcluster) = rclusters.get(&p) {
                    rcluster
                } else {
                    needs_intrinsics_fill_at &= !mask;
                    continue;
                };

                let state = *preloaded_clusters_states
                    .get(&p)
                    .unwrap_or(&ClusterState::Initial);

                if state != ClusterState::Loaded && !state.is_empty_or_occluded() {
                    ready_to_fill = false;
                    break;
                }

                let fill_occluder = if state.is_empty_or_occluded() {
                    // If the neighbour cluster is fully occluded or empty,
                    // clear intrinsics of `cluster` with respective occluders.
                    Some(Occluder::all(rcluster.occluded.load(MO_RELAXED)))
                } else {
                    None
                };

                fill_neighbours.push((
                    p,
                    Arc::clone(&oclusters[&p]),
                    Arc::clone(&rclusters[&p].mesh_can_be_updated),
                    fill_occluder,
                ));
            }

            if !ready_to_fill {
                continue;
            }

            self.curr_clusters_intrinsics_updating_n.fetch_add(1, MO_RELAXED);

            let mut rcluster = rclusters.get_mut(pos).unwrap();
            rcluster.needs_intrinsics_clear_at = 0;
            rcluster.needs_intrinsics_fill_at = 0;
            rcluster
                .updating_outer_intrinsics
                .store(needs_intrinsics_fill_at, MO_RELAXED);

            // Subtract fill sides from clear sides because sides for clearing will be overridden by filling
            let clear_mask = needs_intrinsics_clear_at & !needs_intrinsics_fill_at;

            let clear_neighbour_sides: SmallVec<[ClusterPos; 26]> = get_side_clusters(&pos)
                .into_iter()
                .filter_map(|p| {
                    let mask = 1 << neighbour_dir_index(&pos, &p);
                    if (clear_mask & mask) != 0 {
                        Some(p)
                    } else {
                        None
                    }
                })
                .collect();

            let ocluster = Arc::clone(ocluster);
            let pos = *pos;
            let mesh_can_be_updated = Arc::clone(&rcluster.mesh_can_be_updated);
            let curr_updating_clusters_n = Arc::clone(&self.curr_clusters_intrinsics_updating_n);

            // Remove unnecessary filling of neighbour cluster
            // (neighbour cluster is filled mutually with current cluster)
            for (neighbour_pos, _, _, _) in &fill_neighbours {
                let rcluster = rclusters.get_mut(neighbour_pos).unwrap();
                let mask = 1 << neighbour_dir_index(&neighbour_pos, &pos);

                rcluster.needs_intrinsics_fill_at &= !mask;
            }

            intensive_queue().spawn(move || {
                for neighbour_pos in clear_neighbour_sides {
                    let mut cluster = cluster.write();
                    let cluster = unwrap_option!(cluster.as_mut(), continue);
                    let offset = neighbour_pos.get() - pos.get();
                    cluster
                        .raw
                        .clear_outer_intrinsics(glm::convert(offset), Default::default());
                }
                for (neighbour_pos, neighbour, neighbour_mesh_can_be_updated, offloaded_fill_occluder) in
                    fill_neighbours
                {
                    // Use loop in case of failure to acquire a lock of one of the clusters
                    loop {
                        let neighbour_cluster = neighbour.cluster.try_upgradable_read();
                        let cluster = cluster.try_write();

                        if neighbour_cluster.is_none() || cluster.is_none() {
                            continue;
                        }

                        let neighbour_cluster_guard = neighbour_cluster.unwrap();
                        let mut cluster_guard = cluster.unwrap();

                        let cluster = unwrap_option!(cluster_guard.as_mut(), break);

                        if neighbour.state() == ClusterState::Loaded {
                            let offset: I32Vec3 = glm::convert(neighbour_pos.get() - pos.get());
                            let neighbour_cluster = unwrap_option!(neighbour_cluster_guard.as_ref(), break);

                            // Mutually update `cluster` and `neighbour_cluster` intrinsics.
                            // First, propagate lighting in both clusters,
                            // and only then update intrinsics (which are also related by lighting).

                            let dirty_parts = cluster
                                .raw
                                .propagate_outer_lighting(&neighbour_cluster.raw, offset);
                            cluster.dirty_parts |= dirty_parts;

                            // Switch cluster to read, neighbour_cluster to write
                            let cluster_guard = RwLockWriteGuard::downgrade_to_upgradable(cluster_guard);
                            let mut neighbour_cluster_guard =
                                RwLockUpgradableReadGuard::upgrade(neighbour_cluster_guard);
                            let cluster = cluster_guard.as_ref().unwrap();
                            let neighbour_cluster = neighbour_cluster_guard.as_mut().unwrap();

                            let dirty_parts = neighbour_cluster
                                .raw
                                .propagate_outer_lighting(&cluster.raw, -offset);
                            neighbour_cluster.dirty_parts |= dirty_parts;

                            // Finally, mutually paste intrinsics.

                            // Paste intrinsics from `cluster` into `neighbour_cluster`
                            neighbour_cluster
                                .raw
                                .paste_outer_intrinsics(&cluster.raw, -offset);

                            // Switch back: cluster to write, neighbour_cluster to read
                            let mut cluster_guard = RwLockUpgradableReadGuard::upgrade(cluster_guard);
                            let neighbour_cluster_guard =
                                RwLockWriteGuard::downgrade_to_upgradable(neighbour_cluster_guard);
                            let cluster = cluster_guard.as_mut().unwrap();
                            let neighbour_cluster = neighbour_cluster_guard.as_ref().unwrap();

                            // Paste intrinsics from `neighbour_cluster` into `cluster`
                            cluster.raw.paste_outer_intrinsics(&neighbour_cluster.raw, offset);

                            // If either cluster has been changed, mark OverworldCluster as changed
                            // to handle it in further updates of OverworldStreamer.
                            if cluster.dirty_parts.is_any() {
                                ocluster.dirty.store(true, MO_RELAXED);
                            }
                            if neighbour_cluster.dirty_parts.is_any() {
                                neighbour.dirty.store(true, MO_RELAXED);
                            }
                        } else if neighbour.state().is_empty_or_occluded() {
                            let offset = neighbour_pos.get() - pos.get();

                            cluster.raw.clear_outer_intrinsics(
                                glm::convert(offset),
                                CellInfo {
                                    entity_id: Default::default(),
                                    block_id: u16::MAX,
                                    occluder: offloaded_fill_occluder.unwrap(),
                                    light_level: Default::default(),
                                    liquid_state: LiquidState::NULL,
                                    active: false,
                                },
                            );
                        }

                        break;
                    }

                    neighbour_mesh_can_be_updated.store(true, MO_RELEASE);
                }

                mesh_can_be_updated.store(true, MO_RELEASE);
                curr_updating_clusters_n.fetch_sub(1, MO_RELEASE);
            });
        }

        // Schedule task of updating clusters vertex meshes
        for (pos, _) in &keys {
            if self.curr_clusters_meshes_updating_n.load(MO_RELAXED) >= max_updating_clusters_in_progress {
                break;
            }

            let rcluster = self.rclusters.get_mut(pos).unwrap();

            if !rcluster.mesh_can_be_updated.load(MO_RELAXED) {
                continue;
            }
            rcluster.mesh_can_be_updated.store(false, MO_RELAXED);

            let cluster = if let Some(ocluster) = oclusters.get(&pos) {
                Arc::clone(&ocluster.cluster)
            } else {
                continue;
            };

            let mesh_changed = Arc::clone(&rcluster.mesh_changed);
            let empty = Arc::clone(&rcluster.empty);
            let edge_intrinsics = Arc::clone(&rcluster.edge_intrinsics);
            let edge_intrinsics_determined = Arc::clone(&rcluster.edge_intrinsics_determined);
            let curr_clusters_meshes_updating_n = Arc::clone(&self.curr_clusters_meshes_updating_n);
            let updating_outer_intrinsics = Arc::clone(&rcluster.updating_outer_intrinsics);
            let meshes = Arc::clone(&rcluster.meshes);
            let device = Arc::clone(&self.device);
            let res_map = Arc::clone(&self.res_map);

            curr_clusters_meshes_updating_n.fetch_add(1, MO_RELAXED);

            intensive_queue().spawn(move || {
                let cluster = cluster.read();
                let cluster = if let Some(cluster) = cluster.as_ref() {
                    cluster
                } else {
                    curr_clusters_meshes_updating_n.fetch_sub(1, MO_RELEASE);
                    return;
                };

                let result = cluster.raw.update_mesh(&device, &res_map);
                *meshes.lock() = result.meshes;

                // Determine edge intrinsics
                {
                    let mut intrinsics = 0_u8;
                    for i in 0..6 {
                        let occluded = cluster.raw.check_edge_fully_occluded(Facing::from_u8(i));
                        intrinsics |= (occluded as u8) << i;
                    }
                    edge_intrinsics.store(intrinsics, MO_RELEASE);
                    edge_intrinsics_determined.store(true, MO_RELEASE);
                }

                // Check if cluster is empty (cluster's empty status is updated in `update_mesh`)
                empty.store(result.empty, MO_RELEASE);
                // Set flag to initiate object's component::VertexMesh change in the scene
                mesh_changed.store(true, MO_RELEASE);

                updating_outer_intrinsics.store(0, MO_RELEASE);
                curr_clusters_meshes_updating_n.fetch_sub(1, MO_RELEASE);
            });
        }

        dirty_clusters_positions
    }

    /// Creates/removes new render entities.
    pub fn update_scene(&mut self, renderer: &mut Renderer) {
        for obj in self.clusters_entities_to_remove.drain(..) {
            renderer.remove_object(&obj);
        }

        // Add components to the new entities
        for pos in &self.clusters_entities_to_add {
            let transform_comp = component::Transform::new(
                glm::convert(*pos.get()),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let render_config_solid = component::MeshRenderConfig::new(self.cluster_mat_pipeline, false);
            let render_config_translucent = component::MeshRenderConfig::new(self.cluster_mat_pipeline, true);

            let entity_solid = renderer.add_object(VertexMeshObject::new(
                transform_comp,
                render_config_solid,
                Default::default(),
            ));
            let entity_translucent = renderer.add_object(VertexMeshObject::new(
                transform_comp,
                render_config_translucent,
                Default::default(),
            ));

            self.rclusters.get_mut(pos).unwrap().entities = RClusterEntities {
                solid: entity_solid,
                translucent: entity_translucent,
            };
        }

        // Update meshes
        for (pos, _) in self.loaded_clusters.read().iter() {
            let rcluster = &self.rclusters[pos];

            if !rcluster.mesh_changed.load(MO_RELAXED) {
                continue;
            }

            let mut ready_to_set_mesh = true;

            for p in get_side_clusters(pos) {
                if let Some(rcl) = self.rclusters.get(&p) {
                    let updating_mask = rcl.updating_outer_intrinsics.load(MO_RELAXED);
                    let mask = 1 << neighbour_dir_index(&p, pos);

                    if updating_mask & mask != 0 {
                        ready_to_set_mesh = false;
                        break;
                    }
                }
            }

            if ready_to_set_mesh && !rcluster.entities.is_null() {
                let meshes = rcluster.meshes.lock();

                *renderer
                    .access_object(rcluster.entities.solid)
                    .get_mut::<component::VertexMesh>()
                    .unwrap() = component::VertexMesh::new(&meshes.solid.raw());

                *renderer
                    .access_object(rcluster.entities.translucent)
                    .get_mut::<component::VertexMesh>()
                    .unwrap() = component::VertexMesh::new(&meshes.transparent.raw());

                rcluster.mesh_changed.store(false, MO_RELAXED);
            }
        }

        self.clusters_entities_to_remove.clear();
        self.clusters_entities_to_add.clear();
    }
}
