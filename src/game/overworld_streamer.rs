use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::{Cluster, Occluder};
use crate::game::overworld::{
    cluster, generator, Overworld, OverworldCluster, CLUSTER_STATE_DISCARDED, CLUSTER_STATE_INITIAL,
    CLUSTER_STATE_LOADED, CLUSTER_STATE_LOADING, CLUSTER_STATE_OFFLOADED_INVISIBLE,
};
use crate::utils::unwrap_option;
use engine::ecs::{component, scene};
use engine::queue::intensive_queue;
use engine::renderer::Renderer;
use engine::utils::{HashMap, HashSet, MO_ACQUIRE, MO_RELAXED, MO_RELEASE};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, U8Vec3, Vec3};
use parking_lot::{RwLock, RwLockUpgradableReadGuard, RwLockWriteGuard};
use smallvec::SmallVec;
use std::collections::hash_map;
use std::collections::hash_map::Entry;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};
use std::sync::Arc;
use std::time::Instant;
use vk_wrapper as vkw;

pub const FORCED_LOAD_RANGE: usize = 128;

struct FillOcclusionMask {
    clear_mask: u32,
    fill_mask: u32,
}

pub struct OverworldStreamer {
    device: Arc<vkw::Device>,
    main_registry: Arc<MainRegistry>,
    cluster_mat_pipeline: u32,
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    overworld_clusters: Arc<RwLock<HashMap<I64Vec3, Arc<OverworldCluster>>>>,
    rclusters: HashMap<I64Vec3, RCluster>,
    cluster_masks_to_fill_occlusion: HashMap<I64Vec3, FillOcclusionMask>,
    clusters_entities_to_remove: Vec<scene::Entity>,
    clusters_entities_to_add: Vec<I64Vec3>,
    curr_loading_clusters_n: Arc<AtomicU32>,
    curr_clusters_occlusions_updating_n: Arc<AtomicU32>,
    curr_clusters_meshes_updating_n: Arc<AtomicU32>,
}

const CLUSTER_ALL_NEIGHBOURS_3X3: u32 = ((1 << 27) - 1) & !(1 << 13);

struct RCluster {
    creation_time: Instant,
    entity: scene::Entity,
    /// Whether this cluster is invisible due to full occlusion by the neighbour clusters
    occluded: AtomicBool,
    empty: Arc<AtomicBool>,
    /// A mask of 3x3 'sides' of outer occlusions which are needed to be cleared
    needs_occlusion_clear_at: u32,
    /// A mask of 3x3 'sides' which are needed to be filled with occlusion of neighbour clusters
    needs_occlusion_fill_at: u32,
    /// Whether a particular Facing of this cluster is fully occluded by another cluster
    edge_occlusion: Arc<AtomicU8>,
    /// Whether `edge_occlusion` has been determined, only then `occluded` can be set
    edge_occlusion_determined: Arc<AtomicBool>,
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
    glm::try_convert(glm::floor(&(pos / cluster::SIZE as f64))).unwrap()
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

fn facing_dir_index(pos: &I64Vec3, target: &I64Vec3) -> u8 {
    let d: U8Vec3 = glm::try_convert(((target - pos) / (cluster::SIZE as i64)).add_scalar(1)).unwrap();
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
            let edge_occluded = ((rcl.edge_occlusion.load(MO_RELAXED) >> (facing.mirror() as u8)) & 1) == 1;
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
        registry: &Arc<MainRegistry>,
        re: &Renderer,
        cluster_mat_pipeline: u32,
        overworld: &Overworld,
    ) -> Self {
        Self {
            device: Arc::clone(re.device()),
            main_registry: Arc::clone(registry),
            cluster_mat_pipeline,
            stream_pos: Default::default(),
            xz_render_distance: 128,
            y_render_distance: 128,
            overworld_clusters: Arc::clone(overworld.loaded_clusters()),
            rclusters: Default::default(),
            cluster_masks_to_fill_occlusion: Default::default(),
            clusters_entities_to_remove: Default::default(),
            clusters_entities_to_add: Default::default(),
            curr_loading_clusters_n: Arc::new(Default::default()),
            curr_clusters_occlusions_updating_n: Default::default(),
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
        // let layout: HashSet<I64Vec3> = [I64Vec3::new(0, 0, 0), I64Vec3::new(64, 0, 0)]
        //     .into_iter()
        //     .collect();
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

        let max_loading_clusters_in_progress = (intensive_queue().current_num_threads() as u32 / 2).max(1);
        let max_updating_clusters_in_progress = max_loading_clusters_in_progress;

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
                        occluded: Default::default(),
                        empty: Default::default(),
                        entity: scene::Entity::NULL,
                        needs_occlusion_clear_at: 0,
                        needs_occlusion_fill_at: CLUSTER_ALL_NEIGHBOURS_3X3,
                        edge_occlusion: Default::default(),
                        edge_occlusion_determined: Default::default(),
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

            // Mask neighbour rclusters of removed rclusters for clearing their outer occlusion
            for removed_pos in removed_rclusters {
                for p in get_side_clusters(&removed_pos) {
                    if let Some(rcluster) = rclusters.get_mut(&p) {
                        rcluster.needs_occlusion_clear_at |= 1 << facing_dir_index(&p, &removed_pos);
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
                let main_registry = Arc::clone(&self.main_registry);
                let ocluster = Arc::clone(&ocluster);
                let device = Arc::clone(&self.device);
                let pos = p.pos;

                ocluster.state.store(CLUSTER_STATE_LOADING, MO_RELAXED);
                curr_loading_clusters_n.fetch_add(1, MO_ACQUIRE);

                intensive_queue().spawn(move || {
                    if ocluster.state() == CLUSTER_STATE_DISCARDED {
                        curr_loading_clusters_n.fetch_sub(1, MO_RELAXED);
                        return;
                    }

                    let mut cluster = Cluster::new(main_registry.registry(), device);
                    generator::generate_cluster(&mut cluster, &main_registry, pos);

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

                // Set a flag for updating the mesh when possible (eg. when outer occlusion is filled)
                rcluster.mesh_can_be_updated.store(true, MO_RELAXED);

                // Set the need to update occlusions of sides of neighbour clusters
                if changed_sides != 0 {
                    // TODO: optimize based on specific changed sides
                    for p in get_side_clusters(pos) {
                        if let Some(rcl) = rclusters.get_mut(&p) {
                            rcl.needs_occlusion_fill_at |= 1 << facing_dir_index(&p, pos);
                        }
                    }
                }
            }

            // Collect necessary clusters to update their outer occlusions
            for pos in oclusters.keys() {
                let rcluster = &rclusters[pos];

                if rcluster.edge_occlusion_determined.load(MO_RELAXED) {
                    rcluster.occluded.store(
                        is_cluster_visibly_occluded(rclusters, *pos, self.stream_pos),
                        MO_RELAXED,
                    );
                }

                let mut rcluster = rclusters.get_mut(pos).unwrap();

                if rcluster.needs_occlusion_fill_at != 0 || rcluster.needs_occlusion_clear_at != 0 {
                    // Do not update mesh when outer occlusion is not filled yet
                    rcluster.mesh_can_be_updated.store(false, MO_RELAXED);
                }
            }

            // Schedule task of updating clusters outer occlusions
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
                if self.curr_clusters_occlusions_updating_n.load(MO_RELAXED)
                    >= max_updating_clusters_in_progress
                {
                    break;
                }

                let rcluster = unwrap_option!(rclusters.get(pos), continue);
                let mut needs_occlusion_fill_at = rcluster.needs_occlusion_fill_at;
                let mut needs_occlusion_clear_at = rcluster.needs_occlusion_clear_at;

                if needs_occlusion_fill_at == 0 && needs_occlusion_clear_at == 0 {
                    continue;
                }

                let ocluster = unwrap_option!(oclusters.get(pos), continue);

                if ocluster.state() != CLUSTER_STATE_LOADED {
                    continue;
                }
                let cluster = Arc::clone(&ocluster.cluster);

                let mut ready_to_fill = true;
                let mut fill_neighbours = SmallVec::<[(I64Vec3, Arc<_>); 26]>::new();

                for p in get_side_clusters(&pos) {
                    let mask = 1 << facing_dir_index(pos, &p);

                    if !rclusters.contains_key(&p) {
                        needs_occlusion_fill_at &= !mask;
                        continue;
                    }

                    let state = *preloaded_clusters_states
                        .get(&p)
                        .unwrap_or(&CLUSTER_STATE_INITIAL);

                    if state != CLUSTER_STATE_LOADED && state != CLUSTER_STATE_OFFLOADED_INVISIBLE {
                        ready_to_fill = false;
                        break;
                    }

                    fill_neighbours.push((p, Arc::clone(&oclusters[&p])));
                }

                if !ready_to_fill {
                    continue;
                }

                self.curr_clusters_occlusions_updating_n.fetch_add(1, MO_RELAXED);

                let mut rcluster = rclusters.get_mut(pos).unwrap();
                rcluster.needs_occlusion_clear_at = 0;
                rcluster.needs_occlusion_fill_at = 0;

                let mut fill_mask = needs_occlusion_fill_at;
                // Subtract fill sides from clear sides because sides for clearing will be overridden by filling
                let mut clear_mask = needs_occlusion_clear_at & !fill_mask;

                let clear_neighbour_sides: SmallVec<[I64Vec3; 26]> = get_side_clusters(&pos)
                    .into_iter()
                    .filter_map(|p| {
                        let mask = 1 << facing_dir_index(&pos, &p);
                        if (clear_mask & mask) != 0 {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .collect();

                let pos = *pos;
                let mesh_can_be_updated = Arc::clone(&rcluster.mesh_can_be_updated);
                let curr_updating_clusters_n = Arc::clone(&self.curr_clusters_occlusions_updating_n);

                intensive_queue().spawn(move || {
                    for neighbour_pos in clear_neighbour_sides {
                        let cluster = cluster.read();
                        let cluster = unwrap_option!(cluster.as_ref(), continue);
                        let offset = neighbour_pos - pos;
                        cluster.clear_outer_side_occlusion(glm::convert(offset), Occluder::default());
                    }
                    for (neighbour_pos, neighbour) in fill_neighbours {
                        // Use loop in case of failure to acquire a lock of one of the clusters
                        loop {
                            let neighbour_cluster = neighbour.cluster.try_read();
                            // We can use shared (read) access here, but instead use exclusive
                            // to prevent deadlocks of the mutex inside the clusters (interior mutability)
                            let cluster = cluster.try_write();

                            if neighbour_cluster.is_none() || cluster.is_none() {
                                continue;
                            }

                            let neighbour_cluster = neighbour_cluster.unwrap();
                            let mut cluster = cluster.unwrap();

                            let cluster = unwrap_option!(cluster.as_mut(), break);

                            if neighbour.state() == CLUSTER_STATE_LOADED {
                                let offset = neighbour_pos - pos;
                                let neighbour_cluster = unwrap_option!(neighbour_cluster.as_ref(), break);
                                cluster.paste_outer_side_occlusion(&neighbour_cluster, glm::convert(offset));
                            } else if neighbour.state() == CLUSTER_STATE_OFFLOADED_INVISIBLE {
                                let offset = neighbour_pos - pos;
                                cluster.clear_outer_side_occlusion(glm::convert(offset), Occluder::full());
                            }

                            break;
                        }
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
                let edge_occlusion = Arc::clone(&rcluster.edge_occlusion);
                let edge_occlusion_determined = Arc::clone(&rcluster.edge_occlusion_determined);
                let curr_clusters_meshes_updating_n = Arc::clone(&self.curr_clusters_meshes_updating_n);

                curr_clusters_meshes_updating_n.fetch_add(1, MO_RELAXED);

                intensive_queue().spawn(move || {
                    cluster::update_mesh(&cluster);

                    let cluster = cluster.read();
                    let cluster = if let Some(cluster) = cluster.as_ref() {
                        cluster
                    } else {
                        curr_clusters_meshes_updating_n.fetch_sub(1, MO_RELEASE);
                        return;
                    };

                    // Determine edge occlusion
                    {
                        let mut occlusion = 0_u8;
                        for i in 0..6 {
                            let occluded = cluster.check_edge_fully_occluded(Facing::from_u8(i));
                            occlusion |= (occluded as u8) << i;
                        }
                        edge_occlusion.store(occlusion, MO_RELEASE);
                        edge_occlusion_determined.store(true, MO_RELEASE);
                    }

                    // Check if cluster is empty (cluster's empty status is updated in `update_mesh`)
                    empty.store(cluster.is_empty(), MO_RELEASE);
                    // Set a flag to initiate object's component::VertexMesh change in the scene
                    mesh_changed.store(true, MO_RELEASE);

                    curr_clusters_meshes_updating_n.fetch_sub(1, MO_RELEASE);
                });
            }
        }
    }

    /// Creates/removes new render entities.
    pub fn update_renderer(&mut self, re: &Renderer) {
        let scene = re.scene();

        scene.remove_entities(&self.clusters_entities_to_remove);

        // Reserve entities from scene
        let mut entities = scene.create_entities(self.clusters_entities_to_add.len() as u32);

        let mut transform_comps = scene.storage_write::<component::Transform>();
        let mut renderer_comps = scene.storage_write::<component::RenderConfig>();
        let mut vertex_mesh_comps = scene.storage_write::<component::VertexMesh>();

        // Add components to the new entities
        for pos in &self.clusters_entities_to_add {
            let entity = entities.pop().unwrap();
            let transform_comp = component::Transform::new(
                glm::convert(*pos),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let renderer_comp = component::RenderConfig::new(&re, self.cluster_mat_pipeline, false);

            transform_comps.set(entity, transform_comp);
            renderer_comps.set(entity, renderer_comp);

            self.rclusters.get_mut(pos).unwrap().entity = entity;
        }

        // Update meshes
        for (pos, ocluster) in self.overworld_clusters.read().iter() {
            let rcluster = &self.rclusters[pos];
            if rcluster.mesh_changed.swap(false, MO_RELAXED) {
                let cluster = ocluster.cluster.read();
                let cluster = cluster.as_ref().unwrap();
                let mesh = cluster.vertex_mesh();
                if rcluster.entity != scene::Entity::NULL {
                    vertex_mesh_comps.set(rcluster.entity, component::VertexMesh::new(&mesh.raw()));
                }
            }
        }

        self.clusters_entities_to_remove.clear();
        self.clusters_entities_to_add.clear();
    }
}
