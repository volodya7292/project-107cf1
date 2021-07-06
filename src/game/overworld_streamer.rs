use std::collections::hash_map;
use std::sync::atomic::{AtomicBool, AtomicU32};
use std::sync::{atomic, Arc, Mutex};
use std::time::Instant;

use crossbeam_channel as cb;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, Vec3};
use rayon::prelude::*;
use simdnoise::NoiseBuilder;
use smallvec::SmallVec;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::{cluster, OverworldCluster, LOD_LEVELS};
use crate::game::overworld::{generator, Overworld};
use crate::renderer::material_pipeline::MaterialPipeline;
use crate::renderer::{component, Renderer};
use crate::utils::{HashMap, HashSet};
use std::convert::TryInto;
use vk_wrapper::Device;

pub const LOD0_RANGE: usize = 128;

#[derive(Copy, Clone)]
struct ClusterPos {
    level: usize,
    pos: I64Vec3,
}

impl ClusterPos {
    pub fn new(level: usize, pos: I64Vec3) -> ClusterPos {
        ClusterPos { level, pos }
    }
}

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
    mesh_changed: AtomicBool,
}

struct ClusterToRemove {
    level: u32,
    pos: I64Vec3,
    entity: u32,
}

pub struct OverworldStreamer {
    registry: Arc<MainRegistry>,
    renderer: Arc<Mutex<Renderer>>,
    device: Arc<Device>,
    cluster_mat_pipeline: Arc<MaterialPipeline>,
    /// render distance in discrete meters
    render_distance: u32,
    stream_pos: DVec3,
    clusters: [HashMap<I64Vec3, RenderCluster>; LOD_LEVELS],
    side_occlusion_work: SideOcclusionWorkSync,
    clusters_to_remove: Vec<ClusterToRemove>,
    clusters_to_add: Vec<(u32, I64Vec3)>,
    clusters_in_process: Arc<AtomicU32>,
}

pub trait ClusterProvider {}

pub fn cluster_size(level: u32) -> u64 {
    2_u64.pow(6 + level)
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

    fn find_cluster_side_pairs(
        &self,
        overworld: &Overworld,
        cluster_pos: ClusterPos,
    ) -> SmallVec<[ClusterSidePair; 24]> {
        let level = cluster_pos.level;
        let pos = cluster_pos.pos;
        let mut neighbours = SmallVec::<[ClusterSidePair; 24]>::new();

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

            for &d in &Facing::DIRECTIONS {
                for k in 0..2 {
                    for l in 0..2 {
                        let pos2 = pos + map_dst_pos(&glm::convert(d), k, l) * cluster_size0;

                        if self.clusters[level - 1].contains_key(&pos2) {
                            neighbours.push(ClusterSidePair(cluster_pos, ClusterPos::new(level - 1, pos2)));
                        }
                    }
                }
            }
        }

        // Current level
        {
            for &d in &Facing::DIRECTIONS {
                let pos2 = pos + glm::convert::<I32Vec3, I64Vec3>(d) * cluster_size1;

                if overworld.loaded_clusters[level].contains_key(&pos2) {
                    neighbours.push(ClusterSidePair(cluster_pos, ClusterPos::new(level, pos2)));
                }
            }
        }

        // Higher level
        if level < LOD_LEVELS {
            for &d in &Facing::DIRECTIONS {
                let d: I64Vec3 = glm::convert(d);
                let pos2 = pos
                    + d * cluster_size2
                    + d.zip_map(&pos, |d, p| {
                        (-cluster_size1 * (d > 0) as i64) - (p % cluster_size2 * (d == 0) as i64).abs()
                    });

                if glm::all(&pos2.map(|v| v % cluster_size2 != 0)) {
                    continue;
                }

                if overworld.loaded_clusters[level + 1].contains_key(&pos2) {
                    neighbours.push(ClusterSidePair(cluster_pos, ClusterPos::new(level + 1, pos2)));
                }
            }
        }

        neighbours
    }

    fn cluster_update_worker(&self, overworld: &Overworld) {
        let sync = &self.side_occlusion_work;

        while sync.process_count.load(atomic::Ordering::Relaxed) > 0 {
            if let Ok(pair) = sync.receiver.try_recv() {
                let aval0 = self.clusters[pair.0.level][&pair.0.pos]
                    .available
                    .load(atomic::Ordering::Relaxed);
                let aval1 = self.clusters[pair.1.level][&pair.1.pos]
                    .available
                    .load(atomic::Ordering::Relaxed);

                if !aval0 || !aval1 {
                    sync.process_count.fetch_sub(1, atomic::Ordering::Relaxed);
                    continue;
                }

                let cluster0 = &overworld.loaded_clusters[pair.0.level][&pair.0.pos];
                let cluster1 = &overworld.loaded_clusters[pair.1.level][&pair.1.pos];
                let lock0 = cluster0.cluster.try_lock();
                let lock1 = cluster1.cluster.try_lock();

                if lock0.is_ok() && lock1.is_ok() {
                    let mut side_cluster = lock0.unwrap();
                    let mut cluster = lock1.unwrap();

                    let offset = pair.0.pos - pair.1.pos;
                    cluster.paste_outer_side_occlusion(&side_cluster, glm::convert(offset));
                    let offset = pair.1.pos - pair.0.pos;
                    side_cluster.paste_outer_side_occlusion(&cluster, glm::convert(offset));

                    self.clusters[pair.0.level][&pair.0.pos]
                        .changed
                        .store(true, atomic::Ordering::Relaxed);
                    self.clusters[pair.1.level][&pair.1.pos]
                        .changed
                        .store(true, atomic::Ordering::Relaxed);
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
            let stream_pos_i = cluster_aligned_pos(self.stream_pos, i as u32);
            let r = (R * cluster_size) as f64;

            // Fill mask of used clusters
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        let pos = stream_pos_i + I64Vec3::new(x, y, z).add_scalar(-R);
                        let center = (pos * cluster_size).add_scalar(cluster_size / 2);

                        let dist = glm::distance(&self.stream_pos, &glm::convert(center));

                        if dist <= r && dist <= (self.render_distance as f64) {
                            let p = [(R + x) as usize / 2, (R + y) as usize / 2, (R + z) as usize / 2];
                            masks[i + 1][p[0]][p[1]][p[2]] = true;
                        }
                    }
                }
            }

            let mask0 = &masks[i];
            let mask1 = &masks[i + 1];

            // Calculate new cluster positions
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        if !mask1[x as usize][y as usize][z as usize] {
                            continue;
                        }

                        let pos = stream_pos_i + I64Vec3::new(x, y, z).add_scalar(-R) * 2;
                        let in_p = [x * 2 - R, y * 2 - R, z * 2 - R];

                        if (in_p[0] < 0 || in_p[0] >= D)
                            || (in_p[1] < 0 || in_p[1] >= D)
                            || (in_p[2] < 0 || in_p[2] >= D)
                        {
                            continue;
                        }

                        for x2 in 0..2_usize {
                            for y2 in 0..2_usize {
                                for z2 in 0..2_usize {
                                    let p = [
                                        in_p[0] as usize + x2,
                                        in_p[1] as usize + y2,
                                        in_p[2] as usize + z2,
                                    ];
                                    if p[0] < D as usize
                                        && p[1] < D as usize
                                        && p[2] < D as usize
                                        && mask0[p[0]][p[1]][p[2]]
                                    {
                                        continue;
                                    }

                                    let pos = pos + I64Vec3::new(x2 as i64, y2 as i64, z2 as i64);
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
                    if (curr_time_secs - overworld_cluster.creation_time_secs) < 5
                        || cluster_layout[i].contains(pos)
                    {
                        true
                    } else {
                        self.clusters_to_remove.push(ClusterToRemove {
                            level: i as u32,
                            pos: *pos,
                            entity: self.clusters[i][pos].entity,
                        });
                        false
                    }
                });

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
                                mesh_changed: AtomicBool::new(false),
                            },
                        );
                        self.clusters_to_add.push((i as u32, pos));
                    }
                }
            }
        }

        // Generate clusters
        {
            let max_clusters_in_process = num_cpus::get().saturating_sub(2).max(1) as u32;

            'l: for (i, level) in overworld.loaded_clusters.iter_mut().enumerate() {
                for (pos, overworld_cluster) in level {
                    let clusters_in_process = Arc::clone(&self.clusters_in_process);
                    let curr_clusters_in_process = clusters_in_process.load(atomic::Ordering::Acquire);

                    if curr_clusters_in_process >= max_clusters_in_process {
                        break 'l;
                    }

                    if !overworld_cluster.generated.load(atomic::Ordering::Acquire) && pos.y == 0 {
                        let ocluster = Arc::clone(overworld_cluster);
                        let entry_size = 2_u32.pow(i as u32);
                        let pos = *pos;

                        ocluster.generating.store(true, atomic::Ordering::Relaxed);
                        clusters_in_process.fetch_add(1, atomic::Ordering::Relaxed);
                        rayon::spawn(move || {
                            let mut cluster = ocluster.cluster.lock().unwrap();
                            generator::generate_cluster(&mut cluster, pos, entry_size);
                            ocluster.generating.store(false, atomic::Ordering::Release);
                            ocluster.generated.store(true, atomic::Ordering::Release);
                            clusters_in_process.fetch_sub(1, atomic::Ordering::Release);
                        });
                    }
                }
            }
        }

        // Generate meshes
        {
            for (i, level) in overworld.loaded_clusters.iter().enumerate() {
                for (pos, ocluster) in level {
                    let rcluster = &self.clusters[i][pos];
                    let available;

                    if let Ok(cluster) = ocluster.cluster.try_lock() {
                        available = !ocluster.generating.load(atomic::Ordering::Relaxed);
                        if available && cluster.changed() {
                            rcluster.changed.store(true, atomic::Ordering::Relaxed);
                            for p in self.find_cluster_side_pairs(overworld, ClusterPos::new(i, *pos)) {
                                self.side_occlusion_work.sender.send(p).unwrap()
                            }
                        }
                    } else {
                        available = false;
                    }
                    rcluster.available.store(available, atomic::Ordering::Relaxed);
                }
            }

            // Parallelize avoiding deadlocks between side clusters
            let side_pair_count = self.side_occlusion_work.sender.len();
            self.side_occlusion_work
                .process_count
                .store(side_pair_count as u32, atomic::Ordering::Relaxed);
            (0..num_cpus::get()).into_par_iter().for_each(|k| {
                self.cluster_update_worker(overworld);
            });

            for (i, level) in overworld.loaded_clusters.iter().enumerate() {
                let rlevel = &self.clusters[i];

                level.par_iter().for_each(|(pos, ocluster)| {
                    let rcluster = &rlevel[pos];
                    if rcluster.changed.load(atomic::Ordering::Relaxed)
                        && rcluster.available.load(atomic::Ordering::Relaxed)
                    {
                        if let Ok(mut cluster) = ocluster.cluster.lock() {
                            cluster.update_mesh();
                            println!("GOV");
                            rcluster.mesh_changed.store(true, atomic::Ordering::Relaxed);
                            rcluster.changed.store(false, atomic::Ordering::Relaxed);
                        }
                    }
                });
            }
        }
    }

    pub fn update_renderer(&mut self, overworld: &mut Overworld) {
        let renderer = self.renderer.lock().unwrap();
        let scene = renderer.scene();

        scene.remove_entities(
            &self
                .clusters_to_remove
                .iter()
                .map(|v| v.entity)
                .collect::<Vec<u32>>(),
        );
        for v in &self.clusters_to_remove {
            self.clusters[v.level as usize].remove(&v.pos);
        }

        let entities = scene.entities();
        let transform_comps = scene.storage::<component::Transform>();
        let renderer_comps = scene.storage::<component::Renderer>();
        let vertex_mesh_comps = scene.storage::<component::VertexMesh>();
        let children_comps = scene.storage::<component::Children>();

        let mut entities = entities.lock().unwrap();
        let mut transform_comps = transform_comps.write().unwrap();
        let renderer_comps = renderer_comps.write().unwrap();
        let vertex_mesh_comps = vertex_mesh_comps.write().unwrap();
        let children_comps = children_comps.write().unwrap();

        let mut d = cluster::UpdateSystemData {
            mat_pipeline: Arc::clone(&self.cluster_mat_pipeline),
            entities: &mut entities,
            transform: transform_comps,
            renderer: renderer_comps,
            vertex_mesh: vertex_mesh_comps,
            children: children_comps,
        };

        for (level, pos) in &self.clusters_to_add {
            let transform_comp = component::Transform::new(
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );

            let entity = d.entities.create();
            d.transform.set(entity, transform_comp);
            self.clusters[*level as usize].get_mut(&pos).unwrap().entity = entity;
        }

        for (i, level) in overworld.loaded_clusters.iter().enumerate() {
            for (pos, overworld_cluster) in level {
                let render_cluster = &self.clusters[i][pos];
                if render_cluster.mesh_changed.swap(false, atomic::Ordering::Relaxed) {
                    overworld_cluster
                        .cluster
                        .lock()
                        .unwrap()
                        .update_renderable(render_cluster.entity, &mut d);
                }
            }
        }
    }
}

pub fn new(
    registry: &Arc<MainRegistry>,
    renderer: &Arc<Mutex<Renderer>>,
    cluster_mat_pipeline: &Arc<MaterialPipeline>,
) -> OverworldStreamer {
    let (occ_s, occ_r) = cb::unbounded();

    OverworldStreamer {
        registry: Arc::clone(registry),
        renderer: Arc::clone(renderer),
        device: Arc::clone(renderer.lock().unwrap().device()),
        cluster_mat_pipeline: Arc::clone(cluster_mat_pipeline),
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
