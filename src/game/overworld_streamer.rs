use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::{Cluster, IntrinsicBlockData};
use crate::game::overworld::generator::OverworldGenerator;
use crate::game::overworld::occluder::Occluder;
use crate::game::overworld::{
    cluster, generator, Overworld, OverworldCluster, CLUSTER_STATE_DISCARDED, CLUSTER_STATE_INITIAL,
    CLUSTER_STATE_LOADED, CLUSTER_STATE_LOADING, CLUSTER_STATE_OFFLOADED_INVISIBLE,
};
use engine::ecs::scene::Scene;
use engine::ecs::{component, scene};
use engine::queue::intensive_queue;
use engine::renderer::Renderer;
use engine::unwrap_option;
use engine::utils::{HashMap, HashSet, MO_ACQUIRE, MO_RELAXED, MO_RELEASE};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, U8Vec3, Vec3};
use parking_lot::{RwLock, RwLockUpgradableReadGuard, RwLockWriteGuard};
use smallvec::SmallVec;
use std::collections::hash_map;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};
use std::sync::Arc;
use std::time::Instant;
use vk_wrapper as vkw;

pub const FORCED_LOAD_RANGE: usize = 128;

pub struct OverworldStreamer {
    device: Arc<vkw::Device>,
    cluster_mat_pipeline: u32,
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    overworld_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
    overworld_generator: Arc<OverworldGenerator>,
    rclusters: HashMap<I64Vec3, RCluster>,
    clusters_entities_to_remove: Vec<scene::Entity>,
    clusters_entities_to_add: Vec<I64Vec3>,
    curr_loading_clusters_n: Arc<AtomicU32>,
    curr_clusters_intrinsics_updating_n: Arc<AtomicU32>,
    curr_clusters_meshes_updating_n: Arc<AtomicU32>,
}

const CLUSTER_ALL_NEIGHBOURS_3X3: u32 = ((1 << 27) - 1) & !(1 << 13);

struct RCluster {
    creation_time: Instant,
    entity: scene::Entity,
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
}

impl RCluster {
    fn is_visible(&self) -> bool {
        !self.empty.load(MO_RELAXED) && !self.occluded.load(MO_RELAXED)
    }
}

struct ClusterPosD {
    pos: I64Vec3,
    distance: f64,
}

fn cluster_aligned_pos(pos: DVec3) -> I64Vec3 {
    glm::convert_unchecked(glm::floor(&(pos / cluster::SIZE as f64)))
}

fn calc_cluster_layout(
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
) -> HashSet<I64Vec3> {
    let cr = (xz_render_distance / cluster::SIZE as u64 / 2) as i64;
    let cl_size = cluster::SIZE as i64;
    let c_stream_pos = cluster_aligned_pos(stream_pos);
    let mut layout = HashSet::with_capacity((cr * 2 + 1).pow(3) as usize);

    for x in -cr..(cr + 1) {
        for y in -cr..(cr + 1) {
            for z in -cr..(cr + 1) {
                let cp = c_stream_pos + glm::vec3(x, y, z);
                let center = (cp * cl_size).add_scalar(cl_size / 2);
                let d = stream_pos - glm::convert::<I64Vec3, DVec3>(center);

                if (d.x / xz_render_distance as f64).powi(2)
                    + (d.y / y_render_distance as f64).powi(2)
                    + (d.z / xz_render_distance as f64).powi(2)
                    <= 1.0
                {
                    layout.insert(cp * cl_size);
                }
            }
        }
    }

    layout
}

fn get_side_clusters(pos: &I64Vec3) -> SmallVec<[I64Vec3; 26]> {
    let mut neighbours = SmallVec::<[I64Vec3; 26]>::new();
    let cl_size = cluster::SIZE as i64;

    for x in -1..2 {
        for y in -1..2 {
            for z in -1..2 {
                if x == 0 && y == 0 && z == 0 {
                    continue;
                }
                let pos2 = pos + I64Vec3::new(x, y, z) * cl_size;
                neighbours.push(pos2);
            }
        }
    }

    neighbours
}

fn get_side_cluster_by_facing(pos: I64Vec3, facing: Facing) -> I64Vec3 {
    let dir = facing.direction();
    let cl_size = cluster::SIZE as i64;

    pos + glm::convert::<I32Vec3, I64Vec3>(dir) * cl_size
}

fn neighbour_dir_index(pos: &I64Vec3, target: &I64Vec3) -> u8 {
    let d: U8Vec3 = glm::convert_unchecked(((target - pos) / (cluster::SIZE as i64)).add_scalar(1));
    d.x * 9 + d.y * 3 + d.z
}

fn look_cube_directions(dir: DVec3) -> [I32Vec3; 3] {
    [
        I32Vec3::new(-dir.x.signum() as i32, 0, 0),
        I32Vec3::new(0, -dir.y.signum() as i32, 0),
        I32Vec3::new(0, 0, -dir.z.signum() as i32),
    ]
}

fn is_cluster_in_forced_load_range(pos: &I64Vec3, stream_pos: &DVec3) -> bool {
    let center_pos: DVec3 = glm::convert(pos.add_scalar(cluster::SIZE as i64 / 2));
    let dir_to_cluster_unnorm = center_pos - stream_pos;

    dir_to_cluster_unnorm.magnitude() <= FORCED_LOAD_RANGE as f64
}

/// Checks if cluster at `pos` is occluded at all sides by the neighbour clusters
fn is_cluster_visibly_occluded(
    rclusters: &HashMap<I64Vec3, RCluster>,
    pos: I64Vec3,
    stream_pos: DVec3,
) -> bool {
    let center_pos: DVec3 = glm::convert(pos.add_scalar(cluster::SIZE as i64 / 2));
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

    pub fn new(re: &Renderer, cluster_mat_pipeline: u32, overworld: &Overworld) -> Self {
        Self {
            device: Arc::clone(re.device()),
            cluster_mat_pipeline,
            stream_pos: Default::default(),
            xz_render_distance: 128,
            y_render_distance: 128,
            overworld_clusters: Arc::clone(overworld.loaded_clusters()),
            overworld_generator: Arc::clone(overworld.generator()),
            rclusters: Default::default(),
            clusters_entities_to_remove: Default::default(),
            clusters_entities_to_add: Default::default(),
            curr_loading_clusters_n: Arc::new(Default::default()),
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

    /// Generates new clusters and their content.
    pub fn update(&mut self) {
        // Note: the work this method schedules on other threads
        // supposed to run for 20 ms in total.
        let layout = calc_cluster_layout(self.stream_pos, self.xz_render_distance, self.y_render_distance);
        let rclusters = &mut self.rclusters;
        let mut oclusters = self.overworld_clusters.upgradable_read();
        let curr_t = Instant::now();

        let mut sorted_layout: Vec<_> = layout
            .iter()
            .map(|p| ClusterPosD {
                pos: *p,
                distance: (glm::convert::<I64Vec3, DVec3>(p.add_scalar(cluster::SIZE as i64 / 2))
                    - self.stream_pos)
                    .magnitude(),
            })
            .collect();
        sorted_layout.sort_unstable_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

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

                if !do_preserve && rcl.entity != scene::Entity::NULL {
                    self.clusters_entities_to_remove.push(rcl.entity);
                    removed_rclusters.push(*p);
                }

                do_preserve
            });

            let mut oclusters_write = RwLockUpgradableReadGuard::upgrade(oclusters);
            oclusters_write.retain(|p, ocluster| {
                let in_layout = rclusters.contains_key(p);

                if in_layout {
                    let rcl = rclusters.get_mut(p).unwrap();
                    // Do not remove nearby empty/occluded clusters up to FORCED_LOAD_RANGE
                    // to allow fast access to blocks near the streaming position.
                    let visible = rcl.is_visible() || is_cluster_in_forced_load_range(p, &self.stream_pos);

                    if !visible {
                        if rcl.entity != scene::Entity::NULL {
                            self.clusters_entities_to_remove.push(rcl.entity);
                            rcl.entity = scene::Entity::NULL;
                        }
                        *ocluster.cluster.write() = None;
                        ocluster
                            .state
                            .store(CLUSTER_STATE_OFFLOADED_INVISIBLE, MO_RELAXED);
                    }
                } else {
                    ocluster.state.store(CLUSTER_STATE_DISCARDED, MO_RELAXED);
                }

                in_layout
            });

            for p in &layout {
                if let hash_map::Entry::Vacant(e) = rclusters.entry(*p) {
                    e.insert(RCluster {
                        creation_time: curr_t,
                        entity: scene::Entity::NULL,
                        updating_outer_intrinsics: Default::default(),
                        occluded: Default::default(),
                        empty: Default::default(),
                        needs_intrinsics_clear_at: 0,
                        needs_intrinsics_fill_at: CLUSTER_ALL_NEIGHBOURS_3X3,
                        edge_intrinsics: Default::default(),
                        edge_intrinsics_determined: Default::default(),
                        mesh_can_be_updated: Default::default(),
                        mesh_changed: Default::default(),
                    });
                    self.clusters_entities_to_add.push(*p);
                }
                if let hash_map::Entry::Vacant(e) = oclusters_write.entry(*p) {
                    e.insert(Arc::new(OverworldCluster {
                        cluster: Default::default(),
                        state: AtomicU8::new(CLUSTER_STATE_INITIAL),
                        changed: Default::default(),
                    }));
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
                if !rclusters[&p.pos].is_visible() {
                    continue;
                }

                let ocluster = &oclusters[&p.pos];
                if matches!(ocluster.state(), CLUSTER_STATE_LOADED | CLUSTER_STATE_LOADING) {
                    continue;
                }
                ocluster.state.store(CLUSTER_STATE_LOADING, MO_RELAXED);

                let curr_loading_clusters_n = Arc::clone(&self.curr_loading_clusters_n);
                let generator = Arc::clone(&self.overworld_generator);
                let ocluster = Arc::clone(&ocluster);
                let pos = p.pos;

                ocluster.state.store(CLUSTER_STATE_LOADING, MO_RELAXED);
                curr_loading_clusters_n.fetch_add(1, MO_ACQUIRE);

                intensive_queue().spawn(move || {
                    if ocluster.state() == CLUSTER_STATE_DISCARDED {
                        curr_loading_clusters_n.fetch_sub(1, MO_RELAXED);
                        return;
                    }

                    let mut cluster = generator.create_cluster();

                    generator.generate_cluster(&mut cluster, pos);

                    *ocluster.cluster.write() = Some(cluster);

                    // Note: use CAS to account for DISCARDED state
                    let _ = ocluster.state.compare_exchange(
                        CLUSTER_STATE_LOADING,
                        CLUSTER_STATE_LOADED,
                        MO_RELEASE,
                        MO_RELAXED,
                    );
                    ocluster.changed.store(true, MO_RELEASE);

                    curr_loading_clusters_n.fetch_sub(1, MO_RELEASE);
                });
            }
        }

        let preloaded_clusters_states: HashMap<_, _> =
            oclusters.iter().map(|(p, v)| (*p, v.state())).collect();

        // Generate meshes
        {
            // Mark changes
            for (pos, ocluster) in oclusters.iter() {
                if preloaded_clusters_states[pos] != CLUSTER_STATE_LOADED
                    || !ocluster.changed.load(MO_RELAXED)
                {
                    continue;
                }
                ocluster.changed.store(false, MO_RELAXED);

                let rcluster = rclusters.get_mut(pos).unwrap();
                let changed_sides = ocluster.cluster.write().as_mut().unwrap().acquire_changed_sides();

                // Set a flag for updating the mesh when possible (eg. when outer intrinsics is filled)
                rcluster.mesh_can_be_updated.store(true, MO_RELAXED);

                // Set the need to update intrinsics of sides of neighbour clusters
                for (i, side) in changed_sides.iter().enumerate() {
                    if !side {
                        continue;
                    }
                    let changed_dir: I64Vec3 = glm::convert(cluster::neighbour_index_to_dir(i));

                    let p0 = pos / (cluster::SIZE as i64);
                    let p1 = p0 + changed_dir;

                    let min = p0.inf(&p1);
                    let max = p0.sup(&p1);

                    for x in min.x..=max.x {
                        for y in min.y..=max.y {
                            for z in min.z..=max.z {
                                let p = glm::vec3(x, y, z) * (cluster::SIZE as i64);
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

            // Collect necessary clusters to update their outer intrinsics
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
                    // Do not update mesh when outer intrinsics is not filled yet
                    rcluster.mesh_can_be_updated.store(false, MO_RELAXED);
                }
            }

            // Schedule task of updating clusters outer intrinsics
            let mut keys: Vec<_> = oclusters
                .keys()
                .map(|v| {
                    (
                        *v,
                        (glm::convert::<I64Vec3, DVec3>(*v) - self.stream_pos).magnitude_squared(),
                    )
                })
                .collect();
            keys.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for (pos, _) in &keys {
                if self.curr_clusters_intrinsics_updating_n.load(MO_RELAXED)
                    >= max_updating_clusters_in_progress
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

                if ocluster.state() != CLUSTER_STATE_LOADED {
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
                    if !rclusters.contains_key(&p) {
                        needs_intrinsics_fill_at &= !mask;
                        continue;
                    }

                    let state = *preloaded_clusters_states
                        .get(&p)
                        .unwrap_or(&CLUSTER_STATE_INITIAL);

                    if state != CLUSTER_STATE_LOADED && state != CLUSTER_STATE_OFFLOADED_INVISIBLE {
                        ready_to_fill = false;
                        break;
                    }

                    fill_neighbours.push((
                        p,
                        Arc::clone(&oclusters[&p]),
                        Arc::clone(&rclusters[&p].mesh_can_be_updated),
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

                let clear_neighbour_sides: SmallVec<[I64Vec3; 26]> = get_side_clusters(&pos)
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
                for (neighbour_pos, _, _) in &fill_neighbours {
                    let rcluster = rclusters.get_mut(neighbour_pos).unwrap();
                    let mask = 1 << neighbour_dir_index(&neighbour_pos, &pos);

                    rcluster.needs_intrinsics_fill_at &= !mask;
                }

                intensive_queue().spawn(move || {
                    for neighbour_pos in clear_neighbour_sides {
                        let mut cluster = cluster.write();
                        let cluster = unwrap_option!(cluster.as_mut(), continue);
                        let offset = neighbour_pos - pos;
                        cluster.clear_outer_intrinsics(glm::convert(offset), Default::default());
                    }
                    for (neighbour_pos, neighbour, neighbour_mesh_can_be_updated) in fill_neighbours {
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

                            if neighbour.state() == CLUSTER_STATE_LOADED {
                                let offset: I32Vec3 = glm::convert(neighbour_pos - pos);
                                let neighbour_cluster =
                                    unwrap_option!(neighbour_cluster_guard.as_ref(), break);

                                // Mutually update `cluster` and `neighbour_cluster` intrinsics.
                                // First, propagate lighting in both clusters,
                                // and only then update intrinsics (which are also related by lighting).

                                cluster.propagate_outer_lighting(&neighbour_cluster, offset);

                                // Switch cluster to read, neighbour_cluster to write
                                let cluster_guard = RwLockWriteGuard::downgrade_to_upgradable(cluster_guard);
                                let mut neighbour_cluster_guard =
                                    RwLockUpgradableReadGuard::upgrade(neighbour_cluster_guard);
                                let cluster = cluster_guard.as_ref().unwrap();
                                let neighbour_cluster = neighbour_cluster_guard.as_mut().unwrap();

                                neighbour_cluster.propagate_outer_lighting(&cluster, -offset);

                                // Finally, mutually paste intrinsics.

                                // Paste intrinsics from `cluster` into `neighbour_cluster`
                                neighbour_cluster.paste_outer_intrinsics(&cluster, -offset);

                                // Switch back: cluster to write, neighbour_cluster to read
                                let mut cluster_guard = RwLockUpgradableReadGuard::upgrade(cluster_guard);
                                let neighbour_cluster_guard =
                                    RwLockWriteGuard::downgrade_to_upgradable(neighbour_cluster_guard);
                                let cluster = cluster_guard.as_mut().unwrap();
                                let neighbour_cluster = neighbour_cluster_guard.as_ref().unwrap();

                                // Paste intrinsics from `neighbour_cluster` into `cluster`
                                cluster.paste_outer_intrinsics(&neighbour_cluster, offset);

                                // If either cluster has been changed, mark OverworldCluster as changed
                                // to handle it in further updates of OverworldStreamer.
                                if cluster.get_changed_sides().iter().any(|v| *v) {
                                    ocluster.changed.store(true, MO_RELAXED);
                                }
                                if neighbour_cluster.get_changed_sides().iter().any(|v| *v) {
                                    neighbour.changed.store(true, MO_RELAXED);
                                }
                            } else if neighbour.state() == CLUSTER_STATE_OFFLOADED_INVISIBLE {
                                let offset = neighbour_pos - pos;
                                cluster.clear_outer_intrinsics(
                                    glm::convert(offset),
                                    IntrinsicBlockData {
                                        tex_model_id: u16::MAX,
                                        occluder: Occluder::full(),
                                        light_level: Default::default(),
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
                if self.curr_clusters_meshes_updating_n.load(MO_RELAXED) >= max_updating_clusters_in_progress
                {
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
                let device = Arc::clone(&self.device);

                curr_clusters_meshes_updating_n.fetch_add(1, MO_RELAXED);

                intensive_queue().spawn(move || {
                    let cluster = cluster.read();
                    let cluster = if let Some(cluster) = cluster.as_ref() {
                        cluster
                    } else {
                        curr_clusters_meshes_updating_n.fetch_sub(1, MO_RELEASE);
                        return;
                    };

                    cluster.update_mesh(&device);

                    // Determine edge intrinsics
                    {
                        let mut intrinsics = 0_u8;
                        for i in 0..6 {
                            let occluded = cluster.check_edge_fully_occluded(Facing::from_u8(i));
                            intrinsics |= (occluded as u8) << i;
                        }
                        edge_intrinsics.store(intrinsics, MO_RELEASE);
                        edge_intrinsics_determined.store(true, MO_RELEASE);
                    }

                    // Check if cluster is empty (cluster's empty status is updated in `update_mesh`)
                    empty.store(cluster.is_empty(), MO_RELEASE);
                    // Set a flag to initiate object's component::VertexMesh change in the scene
                    mesh_changed.store(true, MO_RELEASE);

                    updating_outer_intrinsics.store(0, MO_RELEASE);
                    curr_clusters_meshes_updating_n.fetch_sub(1, MO_RELEASE);
                });
            }
        }
    }

    /// Creates/removes new render entities.
    pub fn update_scene(&mut self, scene: &Scene) {
        scene.remove_entities(&self.clusters_entities_to_remove);

        // Reserve entities from scene
        let mut entities = scene.create_entities(self.clusters_entities_to_add.len() as u32);

        let mut transform_comps = scene.storage_write::<component::Transform>();
        let mut render_config_comps = scene.storage_write::<component::MeshRenderConfig>();
        let mut vertex_mesh_comps = scene.storage_write::<component::VertexMesh>();

        // Add components to the new entities
        for pos in &self.clusters_entities_to_add {
            let entity = entities.pop().unwrap();
            let transform_comp = component::Transform::new(
                glm::convert(*pos),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let render_config = component::MeshRenderConfig::new(self.cluster_mat_pipeline, false);

            transform_comps.set(entity, transform_comp);
            render_config_comps.set(entity, render_config);

            self.rclusters.get_mut(pos).unwrap().entity = entity;
        }

        // Update meshes
        for (pos, ocluster) in self.overworld_clusters.read().iter() {
            let rcluster = &self.rclusters[pos];

            if !rcluster.mesh_changed.load(MO_RELAXED) {
                continue;
            }

            let cluster = ocluster.cluster.read();
            let cluster = unwrap_option!(cluster.as_ref(), continue);
            let mesh = cluster.vertex_mesh();
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

            if ready_to_set_mesh && rcluster.entity != scene::Entity::NULL {
                vertex_mesh_comps.set(rcluster.entity, component::VertexMesh::new(&mesh.raw()));
                rcluster.mesh_changed.store(false, MO_RELAXED);
            }
        }

        self.clusters_entities_to_remove.clear();
        self.clusters_entities_to_add.clear();
    }
}
