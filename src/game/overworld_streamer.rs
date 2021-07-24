use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::{cluster, OverworldCluster, LOD_LEVELS};
use crate::game::overworld::{generator, Overworld};
use crate::render_engine::material_pipeline::MaterialPipeline;
use crate::render_engine::{component, RenderEngine};
use crate::utils::{HashMap, HashSet, Integer};
use crossbeam_channel as cb;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, TVec3, Vec3};
use rayon::prelude::*;
use simdnoise::NoiseBuilder;
use smallvec::SmallVec;
use std::collections::hash_map;
use std::convert::TryInto;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{atomic, Arc, Mutex};
use std::time::Instant;
use vk_wrapper::Device;

pub const LOD0_RANGE: usize = 128;

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ClusterPos {
    level: usize,
    pos: I64Vec3,
}

impl ClusterPos {
    pub fn new(level: usize, pos: I64Vec3) -> ClusterPos {
        ClusterPos { level, pos }
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
struct ClusterSidePair(ClusterPos, ClusterPos);

struct SideOcclusionWorkSync {
    sender: cb::Sender<ClusterSidePair>,
    receiver: cb::Receiver<ClusterSidePair>,
    process_count: AtomicU32,
}

struct RenderCluster {
    entity: u32,
    available: AtomicBool,
    changed: AtomicBool,
    // TODO OPTIMIZE: specify which side needs to be cleaned
    needs_occlusion_clean: AtomicBool,
    // TODO OPTIMIZE: specify which side needs to be filled
    needs_occlusion_fill: AtomicBool,
    mesh_changed: AtomicBool,
}

struct ClusterToRemove {
    cluster_pos: ClusterPos,
    entity: u32,
}

pub struct OverworldStreamer {
    registry: Arc<MainRegistry>,
    renderer: Arc<Mutex<RenderEngine>>,
    device: Arc<Device>,
    cluster_mat_pipeline: u32,
    /// render distance in discrete meters
    render_distance: u32,
    stream_pos: DVec3,
    clusters: [HashMap<I64Vec3, RenderCluster>; LOD_LEVELS],
    side_occlusion_work: SideOcclusionWorkSync,
    clusters_to_remove: Vec<ClusterToRemove>,
    clusters_to_add: Vec<ClusterPos>,
    clusters_in_process: Arc<AtomicU32>,
}

pub trait ClusterProvider {}

pub fn cluster_size(level: u32) -> u64 {
    cluster::SIZE as u64 * 2_u64.pow(level)
}

fn cluster_aligned_pos(pos: DVec3, cluster_lod: u32) -> I64Vec3 {
    let cluster_step_size = cluster_size(cluster_lod) as f64;
    glm::try_convert(glm::floor(&(pos / cluster_step_size))).unwrap()
}

impl OverworldStreamer {
    const MIN_RENDER_DISTANCE: u32 = 128;
    const MAX_RENDER_DISTANCE: u32 = 65536;

    pub fn set_stream_pos(&mut self, pos: DVec3) {
        self.stream_pos = pos;
    }

    pub fn set_render_distance(&mut self, render_distance: u32) {
        self.render_distance = render_distance
            .min(Self::MAX_RENDER_DISTANCE)
            .max(Self::MIN_RENDER_DISTANCE);
    }

    fn find_side_clusters(
        &self,
        overworld: &Overworld,
        cluster_pos: ClusterPos,
    ) -> SmallVec<[ClusterPos; 24]> {
        let level = cluster_pos.level;
        let pos = cluster_pos.pos;
        let mut neighbours = SmallVec::<[ClusterPos; 24]>::new();

        let cluster_size1 = cluster_size(level as u32) as i64;
        let cluster_size2 = cluster_size(level as u32 + 1) as i64;

        // Lower level
        if level > 0 {
            fn map_dst_pos(d: &I64Vec3, k: i64, l: i64) -> I64Vec3 {
                let dx = (d.x != 0) as i64;
                let dy = (d.y != 0) as i64;
                let dz = (d.z != 0) as i64;

                let (k, l) = (k + 1, l + 1);
                let x = -1 + k * dy + k * dz + 3 * (d.x > 0) as i64;
                let y = -1 + k * dx + l * dz + 3 * (d.y > 0) as i64;
                let z = -1 + l * dx + l * dy + 3 * (d.z > 0) as i64;

                I64Vec3::new(x, y, z)
            }

            let cluster_size0 = cluster_size(level as u32 - 1) as i64;

            for x in -1..3 {
                for y in -1..3 {
                    for z in -1..3 {
                        if x >= 0 && x < 2 && y >= 0 && y < 2 && z >= 0 && z < 2 {
                            continue;
                        }
                        let pos2 = pos + I64Vec3::new(x, y, z) * cluster_size0;

                        if overworld.loaded_clusters[level - 1].contains_key(&pos2) {
                            neighbours.push(ClusterPos::new(level - 1, pos2));
                        }
                    }
                }
            }
        }

        // Current level
        {
            for x in -1..2 {
                for y in -1..2 {
                    for z in -1..2 {
                        if x == 0 && y == 0 && z == 0 {
                            continue;
                        }
                        let pos2 = pos + I64Vec3::new(x, y, z) * cluster_size1;

                        if overworld.loaded_clusters[level].contains_key(&pos2) {
                            neighbours.push(ClusterPos::new(level, pos2));
                        }
                    }
                }
            }
        }

        // Higher level
        if level + 1 < LOD_LEVELS {
            let align = pos.map(|v| v.rem_euclid(cluster_size2));
            let align_pos = pos - align;
            let align_pos2 = pos + align;

            for x in -1..2 {
                for y in -1..2 {
                    for z in -1..2 {
                        if x == 0 && y == 0 && z == 0 {
                            continue;
                        }
                        let xyz = I64Vec3::new(x, y, z);
                        let pos2 = align_pos + xyz * cluster_size2;

                        if glm::any(&glm::not_equal(
                            &(pos2
                                + align.zip_map(&xyz, |a, v| (v < 0 || (v == 0 && a > 0)) as i64)
                                    * cluster_size2),
                            &align_pos2,
                        )) {
                            continue;
                        }
                        if overworld.loaded_clusters[level + 1].contains_key(&pos2) {
                            neighbours.push(ClusterPos::new(level + 1, pos2));
                        }
                    }
                }
            }
        }

        neighbours
    }

    fn cluster_update_worker(&self, overworld: &Overworld) {
        let sync = &self.side_occlusion_work;

        while sync.process_count.load(atomic::Ordering::Relaxed) > 0 {
            if let Ok(pair) = sync.receiver.try_recv() {
                let r_side_cluster = &self.clusters[pair.1.level][&pair.1.pos];
                if !r_side_cluster.available.load(atomic::Ordering::Relaxed) {
                    r_side_cluster
                        .needs_occlusion_fill
                        .store(true, atomic::Ordering::Relaxed);
                    sync.process_count.fetch_sub(1, atomic::Ordering::Relaxed);
                    continue;
                }

                let cluster0 = &overworld.loaded_clusters[pair.0.level][&pair.0.pos];
                let cluster1 = &overworld.loaded_clusters[pair.1.level][&pair.1.pos];
                let lock0 = cluster0.cluster.try_lock();
                let lock1 = cluster1.cluster.try_lock();

                if lock0.is_ok() && lock1.is_ok() {
                    let mut cluster = lock0.unwrap();
                    let mut side_cluster = lock1.unwrap();

                    let offset = pair.0.pos - pair.1.pos;
                    side_cluster.paste_outer_side_occlusion(&cluster, glm::convert(offset));
                    r_side_cluster.changed.store(true, atomic::Ordering::Relaxed);

                    sync.process_count.fetch_sub(1, atomic::Ordering::Relaxed);
                } else {
                    drop(lock0);
                    drop(lock1);
                    sync.sender.send(pair).unwrap();
                }
            }
        }
    }

    pub fn calc_cluster_layout(&self) -> Vec<HashSet<I64Vec3>> {
        const R: i64 = (LOD0_RANGE / cluster::SIZE) as i64;
        const D: i64 = R * 2 + 1;

        let mut cluster_layout = vec![HashSet::with_capacity(512); LOD_LEVELS];
        let mut masks = [[[[false; D as usize]; D as usize]; D as usize]; LOD_LEVELS + 1];

        for i in 0..LOD_LEVELS {
            let cluster_size = cluster_size(i as u32) as i64;
            let stream_pos_i0 = cluster_aligned_pos(self.stream_pos, i as u32);
            let m = stream_pos_i0.map(|v| v.rem_euclid(2));
            let r = (R * cluster_size) as f64;

            // Fill mask of used clusters
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        let xyz = I64Vec3::new(x, y, z);
                        let pos = stream_pos_i0 + xyz.add_scalar(-R);
                        let center = (pos * cluster_size).add_scalar(cluster_size / 2);
                        let dist = glm::distance(&self.stream_pos, &glm::convert(center));

                        if dist <= r && dist <= (self.render_distance as f64) {
                            let p: TVec3<usize> = glm::try_convert((xyz.add_scalar(R) + m) / 2).unwrap();
                            masks[i + 1][p[0]][p[1]][p[2]] = true;
                        }
                    }
                }
            }

            let stream_pos_i1 = cluster_aligned_pos(self.stream_pos, i as u32 + 1);
            let mask0 = &masks[i];
            let mask1 = &masks[i + 1];

            // Calculate new cluster positions
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        if !mask1[x as usize][y as usize][z as usize] {
                            continue;
                        }

                        let xyz = glm::vec3(x, y, z);
                        let pos = (stream_pos_i1 + xyz.add_scalar(-R)) * 2;
                        let in_p = (xyz * 2).add_scalar(-R) - m;

                        for x2 in 0..2_i64 {
                            for y2 in 0..2_i64 {
                                for z2 in 0..2_i64 {
                                    let xyz2 = glm::vec3(x2, y2, z2);
                                    let p: TVec3<usize> = glm::try_convert(in_p + xyz2).unwrap();

                                    if p[0] < D as usize
                                        && p[1] < D as usize
                                        && p[2] < D as usize
                                        && mask0[p[0]][p[1]][p[2]]
                                    {
                                        continue;
                                    }

                                    let pos = pos + xyz2;
                                    cluster_layout[i].insert(pos);
                                }
                            }
                        }
                    }
                }
            }
        }

        cluster_layout
    }

    pub fn update(&mut self, overworld: &mut Overworld) {
        // Add/remove clusters
        {
            let cluster_layout = self.calc_cluster_layout();
            let curr_time_secs = Instant::now().elapsed().as_secs();

            self.clusters_to_remove.clear();
            self.clusters_to_add.clear();

            for i in 0..LOD_LEVELS {
                // Remove unnecessary clusters
                overworld.loaded_clusters[i].retain(|pos, overworld_cluster| {
                    // TODO: uncomment
                    // if (curr_time_secs - overworld_cluster.creation_time_secs) < 5
                    //     || cluster_layout[i].contains(&(pos / (cluster_size(i as u32) as i64)))
                    if cluster_layout[i].contains(&(pos / (cluster_size(i as u32) as i64))) {
                        true
                    } else {
                        self.clusters_to_remove.push(ClusterToRemove {
                            cluster_pos: ClusterPos { level: i, pos: *pos },
                            entity: self.clusters[i][pos].entity,
                        });
                        false
                    }
                });

                // Set `side_occlusion_changed` flag in affected clusters
                for cluster in &self.clusters_to_remove {
                    for p in self.find_side_clusters(overworld, cluster.cluster_pos) {
                        self.clusters[p.level][&p.pos]
                            .needs_occlusion_clean
                            .store(true, atomic::Ordering::Relaxed);
                    }
                }

                // Add missing clusters
                for pos in &cluster_layout[i] {
                    let node_size = 2_u32.pow(i as u32);
                    let pos = pos * (cluster::SIZE as i64) * (node_size as i64);

                    if let hash_map::Entry::Vacant(entry) = overworld.loaded_clusters[i].entry(pos) {
                        let cluster = cluster::new(&self.registry.registry(), &self.device, node_size);

                        entry.insert(Arc::new(OverworldCluster {
                            cluster: Mutex::new(cluster),
                            creation_time_secs: curr_time_secs,
                            generated: AtomicBool::new(false),
                            generating: AtomicBool::new(false),
                        }));
                        self.clusters[i].insert(
                            pos,
                            RenderCluster {
                                entity: u32::MAX,
                                available: AtomicBool::new(false),
                                changed: AtomicBool::new(false),
                                needs_occlusion_clean: AtomicBool::new(false),
                                needs_occlusion_fill: AtomicBool::new(true),
                                mesh_changed: AtomicBool::new(false),
                            },
                        );
                        self.clusters_to_add.push(ClusterPos::new(i, pos));
                    }
                }
            }
        }

        // Generate clusters
        {
            let max_clusters_in_process = num_cpus::get().saturating_sub(2).max(1) as u32;

            'l: for level in &overworld.loaded_clusters {
                for (pos, overworld_cluster) in level {
                    let clusters_in_process = Arc::clone(&self.clusters_in_process);
                    let curr_clusters_in_process = clusters_in_process.load(atomic::Ordering::Acquire);

                    if curr_clusters_in_process >= max_clusters_in_process {
                        break 'l;
                    }

                    if !overworld_cluster.generated.load(atomic::Ordering::Acquire) {
                        let ocluster = Arc::clone(overworld_cluster);
                        let pos = *pos;
                        let main_registry = Arc::clone(&self.registry);

                        ocluster.generating.store(true, atomic::Ordering::Relaxed);
                        clusters_in_process.fetch_add(1, atomic::Ordering::Relaxed);
                        rayon::spawn(move || {
                            let mut cluster = ocluster.cluster.lock().unwrap();
                            generator::generate_cluster(&mut cluster, &main_registry, pos);
                            ocluster.generating.store(false, atomic::Ordering::Release);
                            ocluster.generated.store(true, atomic::Ordering::Release);
                            clusters_in_process.fetch_sub(1, atomic::Ordering::Release);
                        });
                    }
                }
            }
        }

        // TODO OPTIMIZE: Update cluster meshes only when neighbour clusters are fully loaded
        // TODO OPTIMIZE: to compensate for large number of updates due to neighbour updates.
        // TODO: To do this, firstly fully load all the clusters up to a certain radius.
        // TODO: Secondly, fully update their meshes.
        // TODO: Lastly, repeat the same process for a larger radius until the limit is reached.

        // Generate meshes
        {
            // Collect changed clusters
            let mut occlusion_clusters = HashSet::with_capacity(4096);

            for (i, level) in overworld.loaded_clusters.iter().enumerate() {
                for (pos, ocluster) in level {
                    let rcluster = &self.clusters[i][pos];
                    let available = !ocluster.generating.load(atomic::Ordering::Relaxed);
                    rcluster.available.store(available, atomic::Ordering::Relaxed);
                    if !available {
                        continue;
                    }

                    let cluster = ocluster.cluster.lock().unwrap();
                    let cluster_pos = ClusterPos::new(i, *pos);
                    let cluster_changed = cluster.changed();
                    let occlusion_needs = rcluster
                        .needs_occlusion_fill
                        .swap(false, atomic::Ordering::Relaxed)
                        || rcluster.needs_occlusion_clean.load(atomic::Ordering::Relaxed);

                    if cluster_changed || occlusion_needs {
                        let side_clusters = self.find_side_clusters(overworld, cluster_pos);

                        if cluster_changed {
                            rcluster.changed.store(true, atomic::Ordering::Relaxed);
                            for p in &side_clusters {
                                occlusion_clusters.insert(ClusterSidePair(cluster_pos, *p));
                            }
                        }
                        if occlusion_needs {
                            for p in &side_clusters {
                                occlusion_clusters.insert(ClusterSidePair(*p, cluster_pos));
                            }
                        }
                    }
                }
            }
            for p in occlusion_clusters {
                self.side_occlusion_work.sender.send(p).unwrap();
            }

            // Clean cluster outer side occlusion to not keep old occlusion before filling
            overworld
                .loaded_clusters
                .par_iter()
                .enumerate()
                .for_each(|(i, level)| {
                    level.par_iter().for_each(|(pos, ocluster)| {
                        let rcluster = &self.clusters[i][pos];

                        if rcluster.available.load(atomic::Ordering::Relaxed)
                            && rcluster.needs_occlusion_clean.load(atomic::Ordering::Relaxed)
                        {
                            let mut cluster = ocluster.cluster.lock().unwrap();
                            cluster.clean_outer_side_occlusion();
                            rcluster
                                .needs_occlusion_clean
                                .store(false, atomic::Ordering::Relaxed)
                        }
                    });
                });

            // Update cluster outer side occlusions
            // Note: parallelize avoiding deadlocks between side clusters
            let side_pair_count = self.side_occlusion_work.sender.len();
            self.side_occlusion_work
                .process_count
                .store(side_pair_count as u32, atomic::Ordering::Relaxed);
            (0..num_cpus::get()).into_par_iter().for_each(|k| {
                self.cluster_update_worker(overworld);
            });

            overworld
                .loaded_clusters
                .par_iter()
                .enumerate()
                .for_each(|(i, level)| {
                    let rlevel = &self.clusters[i];

                    level.par_iter().for_each(|(pos, ocluster)| {
                        let rcluster = &rlevel[pos];
                        if rcluster.changed.load(atomic::Ordering::Relaxed)
                            && rcluster.available.load(atomic::Ordering::Relaxed)
                        {
                            let mut cluster = ocluster.cluster.lock().unwrap();
                            cluster.update_mesh();
                            rcluster.mesh_changed.store(true, atomic::Ordering::Relaxed);
                            rcluster.changed.store(false, atomic::Ordering::Relaxed);
                        }
                    });
                });
        }
    }

    pub fn update_renderer(&mut self, overworld: &mut Overworld) {
        let renderer = self.renderer.lock().unwrap();
        let scene = renderer.scene();

        for v in &self.clusters_to_remove {
            self.clusters[v.cluster_pos.level].remove(&v.cluster_pos.pos);
        }
        if !self.clusters_to_remove.is_empty() {
            println!("to remove: {}", self.clusters_to_remove.len());
        }

        // TODO: remove render_engine of a cluster only if lower/higher-lod replacement clusters are already generated
        component::remove_entities(
            scene,
            &self
                .clusters_to_remove
                .iter()
                .map(|v| v.entity)
                .collect::<Vec<u32>>(),
        );

        let entities = scene.entities();
        let transform_comps = scene.storage::<component::Transform>();
        let renderer_comps = scene.storage::<component::Renderer>();
        let vertex_mesh_comps = scene.storage::<component::VertexMesh>();
        let parent_comps = scene.storage::<component::Parent>();
        let children_comps = scene.storage::<component::Children>();

        let mut entities = entities.lock().unwrap();
        let mut transform_comps = transform_comps.write().unwrap();
        let renderer_comps = renderer_comps.write().unwrap();
        let vertex_mesh_comps = vertex_mesh_comps.write().unwrap();
        let parent_comps = parent_comps.write().unwrap();
        let children_comps = children_comps.write().unwrap();

        let mut d = cluster::UpdateSystemData {
            mat_pipeline: self.cluster_mat_pipeline,
            entities: &mut entities,
            transform: transform_comps,
            renderer: renderer_comps,
            vertex_mesh: vertex_mesh_comps,
            parent: parent_comps,
            children: children_comps,
        };

        for cluster_pos in &self.clusters_to_add {
            let pos = &cluster_pos.pos;
            let transform_comp = component::Transform::new(
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );

            let entity = d.entities.create();
            d.transform.set(entity, transform_comp);
            self.clusters[cluster_pos.level].get_mut(pos).unwrap().entity = entity;
        }

        for (i, level) in overworld.loaded_clusters.iter().enumerate() {
            for (pos, overworld_cluster) in level {
                let render_cluster = &self.clusters[i][pos];
                if render_cluster.mesh_changed.swap(false, atomic::Ordering::Relaxed) {
                    overworld_cluster.cluster.lock().unwrap().update_renderable(
                        &renderer,
                        render_cluster.entity,
                        &mut d,
                    );
                }
            }
        }
    }
}

pub fn new(
    registry: &Arc<MainRegistry>,
    renderer: &Arc<Mutex<RenderEngine>>,
    cluster_mat_pipeline: u32,
) -> OverworldStreamer {
    let (occ_s, occ_r) = cb::unbounded();

    OverworldStreamer {
        registry: Arc::clone(registry),
        renderer: Arc::clone(renderer),
        device: Arc::clone(renderer.lock().unwrap().device()),
        cluster_mat_pipeline: cluster_mat_pipeline,
        render_distance: 0,
        stream_pos: DVec3::new(0.0, 0.0, 0.0),
        clusters: Default::default(),
        side_occlusion_work: SideOcclusionWorkSync {
            sender: occ_s,
            receiver: occ_r,
            process_count: Default::default(),
        },
        clusters_to_remove: vec![],
        clusters_to_add: vec![],
        clusters_in_process: Arc::new(AtomicU32::new(0)),
    }
}
