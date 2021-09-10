use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::{cluster, generator, ClusterState, Overworld, OverworldCluster};
use crate::render_engine::{component, scene, RenderEngine};
use crate::utils::{HashMap, HashSet, MO_ACQUIRE, MO_RELAXED, MO_RELEASE};
use crossbeam_channel as cb;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, Mat3, TMat3, U32Vec3, U8Vec3, Vec3};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::hash_map;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};
use std::sync::{atomic, Arc, Mutex, RwLock};
use std::time::Instant;
use vk_wrapper as vkw;

pub const INVISIBLE_LOAD_RANGE: usize = 128;

pub struct OverworldStreamer {
    device: Arc<vkw::Device>,
    main_registry: Arc<MainRegistry>,
    cluster_mat_pipeline: u32,
    re: Arc<Mutex<RenderEngine>>,
    stream_pos: DVec3,
    render_distance: u64,
    overworld: Overworld,
    rclusters: HashMap<I64Vec3, RCluster>,
    clusters_to_remove: Vec<scene::Entity>,
    clusters_to_add: Vec<I64Vec3>,
    curr_loading_clusters_n: Arc<AtomicU32>,
}

struct RCluster {
    creation_time: Instant,
    occluded: AtomicBool,
    empty: AtomicBool,
    entity: scene::Entity,
    changed: AtomicBool,
    needs_occlusion_fill: AtomicU32,
    edge_occlusion: AtomicU8,
    needs_occlusion_check_update: AtomicBool,
    mesh_changed: AtomicBool,
}

struct ClusterPosD {
    pos: I64Vec3,
    distance: f64,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct ClusterSidePair(I64Vec3, I64Vec3);

fn cluster_aligned_pos(pos: DVec3) -> I64Vec3 {
    glm::try_convert(glm::floor(&(pos / cluster::SIZE as f64))).unwrap()
}

fn calc_cluster_layout(stream_pos: DVec3, render_distance: u64) -> HashSet<I64Vec3> {
    let mut layout = HashSet::with_capacity(render_distance.pow(3) as usize);
    let cr = (render_distance / cluster::SIZE as u64 / 2) as i64;
    let cl_size = cluster::SIZE as i64;
    let c_stream_pos = cluster_aligned_pos(stream_pos);

    for x in -cr..(cr + 1) {
        for y in -cr..(cr + 1) {
            for z in -cr..(cr + 1) {
                let cp = c_stream_pos + glm::vec3(x, y, z);
                let center = (cp * cl_size).add_scalar(cl_size / 2);
                let d = glm::distance2(&stream_pos, &glm::convert(center));

                if d <= (render_distance.pow(2) as f64) {
                    layout.insert(cp * cl_size);
                }
            }
        }
    }

    layout
}

fn get_side_clusters(pos: I64Vec3) -> SmallVec<[I64Vec3; 26]> {
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

fn facing_dir_index(pos: I64Vec3, target: I64Vec3) -> u8 {
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

fn check_cluster_vis_occlusion(
    rclusters: &HashMap<I64Vec3, RCluster>,
    pos: I64Vec3,
    stream_pos: DVec3,
) -> bool {
    let center_pos: DVec3 = glm::convert(pos.add_scalar(cluster::SIZE as i64 / 2));
    let dir_to_cluster_unnorm = center_pos - stream_pos;

    if glm::length(&dir_to_cluster_unnorm) <= INVISIBLE_LOAD_RANGE as f64 {
        return false;
    }

    let look_dirs = look_cube_directions(dir_to_cluster_unnorm);
    let mut edges_occluded = true;

    for dir in look_dirs {
        if let Some(facing) = Facing::from_direction(dir) {
            let p = get_side_cluster_by_facing(pos, facing);

            if let Some(rcl) = rclusters.get(&p) {
                let edge_occluded =
                    ((rcl.edge_occlusion.load(MO_RELAXED) >> (facing.mirror() as u8)) & 1) == 1;
                edges_occluded &= edge_occluded;
            } else {
                edges_occluded = false;
                break;
            }

            if !edges_occluded {
                break;
            }
        }
    }

    edges_occluded
}

fn cluster_update_worker(
    process_count: Arc<AtomicU32>,
    receiver: cb::Receiver<ClusterSidePair>,
    sender: cb::Sender<ClusterSidePair>,
    rclusters: &HashMap<I64Vec3, RCluster>,
    oclusters: &HashMap<I64Vec3, Arc<OverworldCluster>>,
) {
    while process_count.load(MO_RELAXED) > 0 {
        if let Ok(pair) = receiver.try_recv() {
            let r_side_cluster = &rclusters[&pair.1];

            let cluster0 = &oclusters[&pair.0];
            let cluster1 = &oclusters[&pair.1];

            let lock0 = cluster0.cluster.try_read();
            let lock1 = cluster1.cluster.try_write();

            if lock0.is_ok() && lock1.is_ok() {
                let cluster = lock0.unwrap();
                let mut side_cluster = lock1.unwrap();

                let offset = pair.0 - pair.1;
                side_cluster.paste_outer_side_occlusion(&cluster, glm::convert(offset));

                r_side_cluster.changed.store(true, MO_RELAXED);
                r_side_cluster
                    .needs_occlusion_fill
                    .fetch_and(!(1 << facing_dir_index(pair.1, pair.0)), MO_RELAXED);

                process_count.fetch_sub(1, MO_RELAXED);
            } else {
                drop(lock0);
                drop(lock1);
                sender.send(pair).unwrap();
            }
        }
    }
}

impl OverworldStreamer {
    const MIN_RENDER_DISTANCE: u64 = 128;
    const MAX_RENDER_DISTANCE: u64 = 1024;

    pub fn new(
        registry: &Arc<MainRegistry>,
        re: &Arc<Mutex<RenderEngine>>,
        cluster_mat_pipeline: u32,
        overworld: Overworld,
    ) -> Self {
        Self {
            device: Arc::clone(re.lock().unwrap().device()),
            main_registry: Arc::clone(registry),
            cluster_mat_pipeline,
            re: Arc::clone(re),
            stream_pos: Default::default(),
            render_distance: 128,
            overworld,
            rclusters: Default::default(),
            clusters_to_remove: Default::default(),
            clusters_to_add: Default::default(),
            curr_loading_clusters_n: Arc::new(Default::default()),
        }
    }

    pub fn set_render_distance(&mut self, dist: u64) {
        self.render_distance = dist.min(Self::MAX_RENDER_DISTANCE).max(Self::MIN_RENDER_DISTANCE);
    }

    pub fn set_stream_pos(&mut self, pos: DVec3) {
        self.stream_pos = pos;
    }

    /// Generates new clusters and their content.
    pub fn update(&mut self) {
        let layout = calc_cluster_layout(self.stream_pos, self.render_distance);
        let curr_t = Instant::now();
        let device = &self.device;
        let stream_pos = self.stream_pos;
        let rclusters = &mut self.rclusters;
        let oclusters = &mut self.overworld.loaded_clusters;
        let clusters_to_remove = &mut self.clusters_to_remove;
        let clusters_to_add = &mut self.clusters_to_add;

        // Add/remove clusters
        {
            rclusters.retain(|p, rcl| {
                let in_layout = layout.contains(p);
                let timeout = (curr_t - rcl.creation_time).as_secs() >= 3;
                let preserve = in_layout || !timeout;

                if !preserve {
                    if rcl.entity != scene::Entity::NULL {
                        clusters_to_remove.push(rcl.entity);
                    }
                }
                preserve
            });
            oclusters.retain(|p, ocl| {
                let in_layout = rclusters.contains_key(p);
                if !in_layout {
                    return false;
                }
                let rcl = rclusters.get_mut(p).unwrap();
                let empty = rcl.empty.load(MO_RELAXED);
                let occluded = rcl.occluded.load(MO_RELAXED);

                if (empty || occluded) && rcl.entity != scene::Entity::NULL {
                    clusters_to_remove.push(rcl.entity);
                    rcl.entity = scene::Entity::NULL;
                }
                !empty && !occluded
            });

            for p in &layout {
                if let hash_map::Entry::Vacant(e) = rclusters.entry(*p) {
                    e.insert(RCluster {
                        creation_time: curr_t,
                        occluded: Default::default(),
                        empty: Default::default(),
                        entity: scene::Entity::NULL,
                        changed: Default::default(),
                        needs_occlusion_fill: AtomicU32::new(((1 << 27) - 1) & !(1 << 13)),
                        edge_occlusion: Default::default(),
                        needs_occlusion_check_update: AtomicBool::new(true),
                        mesh_changed: Default::default(),
                    });
                    clusters_to_add.push(*p);
                }
            }
        }

        let mut sorted_layout: Vec<_> = layout
            .iter()
            .map(|p| ClusterPosD {
                pos: *p,
                distance: glm::distance(&glm::convert(p.add_scalar(cluster::SIZE as i64 / 2)), &stream_pos),
            })
            .collect();
        sorted_layout.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());

        // Generate clusters
        {
            let max_clusters_in_process = (rayon::current_num_threads() - 1).max(1) as u32;
            let registry = self.main_registry.registry();

            for p in &sorted_layout {
                if self.curr_loading_clusters_n.load(MO_ACQUIRE) >= max_clusters_in_process {
                    break;
                }
                if rclusters[&p.pos].empty.load(MO_RELAXED) || rclusters[&p.pos].occluded.load(MO_RELAXED) {
                    continue;
                }

                let ocluster = Arc::clone(oclusters.entry(p.pos).or_insert_with(|| {
                    Arc::new(OverworldCluster {
                        cluster: RwLock::new(Cluster::new(registry, device)),
                        state: AtomicU8::new(ClusterState::UNLOADED as u8),
                    })
                }));
                if matches!(ocluster.state(), ClusterState::LOADED | ClusterState::LOADING) {
                    continue;
                }

                let curr_loading_clusters_n = Arc::clone(&self.curr_loading_clusters_n);
                let main_registry = Arc::clone(&self.main_registry);
                let pos = p.pos;

                ocluster.state.store(ClusterState::LOADING as u8, MO_RELAXED);
                curr_loading_clusters_n.fetch_add(1, MO_ACQUIRE);

                rayon::spawn_fifo(move || {
                    generator::generate_cluster(&mut ocluster.cluster.write().unwrap(), &main_registry, pos);

                    ocluster.state.store(ClusterState::LOADED as u8, MO_RELEASE);
                    curr_loading_clusters_n.fetch_sub(1, MO_RELEASE);
                });
            }
        }

        // Generate meshes
        {
            // Mark changes
            oclusters.par_iter().for_each(|(pos, ocluster)| {
                if ocluster.state() != ClusterState::LOADED {
                    return;
                }
                let rcluster = &rclusters[pos];

                let cluster = ocluster.cluster.read().unwrap();
                let cluster_changed = cluster.changed();
                if !cluster_changed {
                    return;
                }
                rcluster.changed.store(true, MO_RELAXED);

                let changed_sides = cluster.changed_sides();
                if changed_sides != 0 {
                    for p in get_side_clusters(*pos) {
                        if !rclusters.contains_key(&p) {
                            continue;
                        }
                        rclusters[&p]
                            .needs_occlusion_fill
                            .fetch_or(1 << facing_dir_index(p, *pos), MO_RELAXED);
                    }
                }
                if rcluster.needs_occlusion_check_update.swap(false, MO_RELAXED) {
                    let mut edge_occlusion = 0_u8;

                    for i in 0..6 {
                        let occluded = cluster.check_edge_fully_occluded(Facing::from_u8(i));
                        edge_occlusion |= (occluded as u8) << i;
                    }
                    let empty = cluster.is_empty();

                    rcluster.edge_occlusion.store(edge_occlusion, MO_RELAXED);
                    rcluster.empty.store(empty, MO_RELAXED);
                }
            });

            // Collect changes
            let side_occlusion_work = cb::bounded::<ClusterSidePair>(oclusters.len() * 26);

            for (pos, ocluster) in oclusters.iter() {
                if ocluster.state() != ClusterState::LOADED {
                    continue;
                }
                let rcluster = &rclusters[pos];

                rcluster.occluded.store(
                    check_cluster_vis_occlusion(rclusters, *pos, self.stream_pos),
                    MO_RELAXED,
                );

                if rcluster.needs_occlusion_fill.load(MO_RELAXED) == 0 {
                    continue;
                }
                let side_clusters: SmallVec<[_; 26]> = get_side_clusters(*pos)
                    .into_iter()
                    .filter(|p| {
                        let exists = rclusters.contains_key(p)
                            && !rclusters[p].occluded.load(MO_RELAXED)
                            && !rclusters[p].empty.load(MO_RELAXED);
                        if !exists {
                            rcluster
                                .needs_occlusion_fill
                                .fetch_and(!(1 << facing_dir_index(*pos, *p)), MO_RELAXED);
                        }
                        exists
                    })
                    .collect();

                let mut ready_to_update = true;

                for p in &side_clusters {
                    if !oclusters.contains_key(p) || oclusters[p].state() != ClusterState::LOADED {
                        ready_to_update = false;
                        break;
                    }
                }
                if ready_to_update {
                    let mask = rcluster.needs_occlusion_fill.load(MO_RELAXED);

                    for p in side_clusters {
                        if (mask & (1 << facing_dir_index(*pos, p))) != 0 {
                            side_occlusion_work.0.send(ClusterSidePair(p, *pos)).unwrap();
                        }
                    }
                }
            }

            // Update cluster outer side occlusions
            let side_pair_count = side_occlusion_work.0.len();
            let process_count = Arc::new(AtomicU32::new(side_pair_count as u32));
            (0..num_cpus::get()).into_par_iter().for_each(|_| {
                let process_count = Arc::clone(&process_count);
                let receiver = side_occlusion_work.1.clone();
                let sender = side_occlusion_work.0.clone();
                cluster_update_worker(process_count, receiver, sender, rclusters, oclusters);
            });

            oclusters.par_iter().for_each(|(pos, ocluster)| {
                let rcluster = &rclusters[pos];

                if rcluster.changed.load(MO_RELAXED)
                    && ocluster.state() == ClusterState::LOADED
                    && rcluster.needs_occlusion_fill.load(MO_RELAXED) == 0
                {
                    let mut cluster = ocluster.cluster.write().unwrap();

                    cluster.update_mesh();
                    rcluster.mesh_changed.store(true, MO_RELAXED);
                    rcluster.changed.store(false, MO_RELAXED);

                    rcluster.needs_occlusion_check_update.store(true, MO_RELAXED);
                }
            });
        }
    }

    /// Creates/removes new render entities.
    pub fn update_renderer(&mut self) {
        let re = self.re.lock().unwrap();
        let scene = re.scene();

        component::remove_entities(scene, &self.clusters_to_remove);

        // Reserve entities from scene
        let mut entities: Vec<_> = {
            let mut entities = scene.entities().write().unwrap();
            (0..self.clusters_to_add.len())
                .map(|_| entities.create())
                .collect()
        };

        let transform_comps = scene.storage::<component::Transform>();
        let renderer_comps = scene.storage::<component::Renderer>();
        let vertex_mesh_comps = scene.storage::<component::VertexMesh>();
        let mut transform_comps = transform_comps.write();
        let mut renderer_comps = renderer_comps.write();
        let mut vertex_mesh_comps = vertex_mesh_comps.write();

        // Add components to the new entities
        for pos in &self.clusters_to_add {
            let entity = entities.pop().unwrap();
            let transform_comp = component::Transform::new(
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let renderer_comp = component::Renderer::new(&re, self.cluster_mat_pipeline, false);

            transform_comps.set(entity, transform_comp);
            renderer_comps.set(entity, renderer_comp);

            self.rclusters.get_mut(pos).unwrap().entity = entity;
        }

        // Update meshes
        for (pos, ocluster) in &self.overworld.loaded_clusters {
            let rcluster = &self.rclusters[pos];
            if rcluster.mesh_changed.swap(false, MO_RELAXED) {
                let cluster = ocluster.cluster.read().unwrap();
                let mesh = cluster.vertex_mesh();
                vertex_mesh_comps.set(rcluster.entity, component::VertexMesh::new(&mesh.raw()));
                // rcluster.mesh_available.store(true, atomic::Ordering::Relaxed);
            }
        }

        self.clusters_to_remove.clear();
        self.clusters_to_add.clear();
    }
}
