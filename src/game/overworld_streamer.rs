use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::{cluster, OverworldCluster, LOD_LEVELS};
use crate::game::overworld::{generator, Overworld};
use crate::render_engine::{component, scene, RenderEngine};
use crate::utils::{HashMap, HashSet};
use crossbeam_channel as cb;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3, TVec3, Vec3};
use rayon::prelude::*;
use rust_dense_bitset::{BitSet, DenseBitSet};
use smallvec::smallvec;
use smallvec::SmallVec;
use std::collections::hash_map;
use std::convert::TryInto;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8};
use std::sync::{atomic, Arc, Mutex, RwLock};
use std::time::Instant;
use vk_wrapper::Device;

pub const LOD0_RANGE: usize = 256;
pub const INVISIBLE_LOAD_RANGE: usize = 128;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
struct ClusterPos {
    level: usize,
    pos: I64Vec3,
}

impl ClusterPos {
    pub fn new(level: usize, pos: I64Vec3) -> ClusterPos {
        let s = cluster_size(level as u32) as i64;
        assert!(glm::all(&pos.map(|v| v.rem_euclid(s) == 0)));

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

struct AbstractCluster {
    occluded: bool,
    occluded_new: bool,
    occlusion_outdated: bool,
    empty: AtomicBool,
    ready: AtomicBool,
}

struct RenderCluster {
    entity: scene::Entity,
    available: AtomicBool,
    changed: AtomicBool,
    // TODO OPTIMIZE: specify which side needs to be filled
    needs_occlusion_fill: AtomicBool,
    mesh_changed: AtomicBool,
    edge_occlusion: AtomicU8,
    mesh_available: AtomicBool,
}

struct ClusterPosD {
    pos: I64Vec3,
    distance: f64,
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
    aclusters: [HashMap<I64Vec3, AbstractCluster>; LOD_LEVELS],
    side_occlusion_work: SideOcclusionWorkSync,
    clusters_to_remove: HashMap<ClusterPos, bool>,
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

fn look_cube_directions(dir: DVec3) -> [I32Vec3; 3] {
    [
        I32Vec3::new(-dir.x.signum() as i32, 0, 0),
        I32Vec3::new(0, -dir.y.signum() as i32, 0),
        I32Vec3::new(0, 0, -dir.z.signum() as i32),
    ]
}

fn get_side_clusters_by_facing(cluster_pos: ClusterPos, facing: Facing) -> SmallVec<[ClusterPos; 6]> {
    let level = cluster_pos.level;
    let pos = cluster_pos.pos;
    let dir = facing.direction();
    let mut neighbours = SmallVec::<[ClusterPos; 6]>::new();

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

        for k in 0..2 {
            for l in 0..2 {
                let pos2 = pos + map_dst_pos(&glm::convert(dir), k, l) * cluster_size0;
                neighbours.push(ClusterPos::new(level - 1, pos2));
            }
        }
    }

    // Current level
    {
        let pos2 = pos + glm::convert::<I32Vec3, I64Vec3>(dir) * cluster_size1;
        neighbours.push(ClusterPos::new(level, pos2));
    }

    // Higher level
    if level + 1 < LOD_LEVELS {
        let d: I64Vec3 = glm::convert(dir);
        let align = pos.map(|v| v.rem_euclid(cluster_size2));
        let align_pos = pos - align;
        let align_pos2 = pos + align;
        let pos2 = align_pos + d * cluster_size2;

        if glm::all(&glm::equal(
            &(pos2 + align.zip_map(&d, |a, v| (v < 0 || (v == 0 && a > 0)) as i64) * cluster_size2),
            &align_pos2,
        )) {
            neighbours.push(ClusterPos::new(level + 1, pos2));
        }
    }

    neighbours
}

fn check_cluster_vis_occlusion(
    clusters: &[HashMap<I64Vec3, RenderCluster>; LOD_LEVELS],
    aclusters: &[HashMap<I64Vec3, AbstractCluster>; LOD_LEVELS],
    layout: &[HashSet<I64Vec3>; LOD_LEVELS],
    level: u32,
    pos: I64Vec3,
    stream_pos: DVec3,
) -> bool {
    let cluster_size = cluster_size(level) as i64;
    let center_pos: DVec3 = glm::convert(pos.add_scalar(cluster_size / 2));
    let dir_to_cluster_unnorm = center_pos - stream_pos;

    if glm::length(&dir_to_cluster_unnorm) <= INVISIBLE_LOAD_RANGE as f64 {
        return false;
    }

    let look_dirs = look_cube_directions(dir_to_cluster_unnorm);
    let mut edges_occluded = true;

    'l: for dir in look_dirs {
        if let Some(facing) = Facing::from_direction(dir) {
            let sc = get_side_clusters_by_facing(ClusterPos::new(level as usize, pos), facing);

            for p in sc {
                let lpos = p.pos / self::cluster_size(p.level as u32) as i64;

                if !layout[p.level].contains(&lpos) {
                    continue;
                }

                let edge_occluded = if let Some(rcluster) = clusters[p.level].get(&p.pos) {
                    ((rcluster.edge_occlusion.load(atomic::Ordering::Relaxed) >> (facing.mirror() as u8)) & 1)
                        == 1
                } else {
                    aclusters[p.level].get(&p.pos).map_or(false, |acluster| {
                        acluster.occluded && !acluster.occlusion_outdated
                    })
                };
                edges_occluded &= edge_occluded;

                if !edges_occluded {
                    break 'l;
                }
            }
        }
    }

    edges_occluded
}

/// Returns `true` if cluster can be replaced by lower or higher lod clusters or if cluster's replacement is itself.
fn cluster_replacement(
    rclusters: &[HashMap<I64Vec3, RenderCluster>; LOD_LEVELS],
    aclusters: &[HashMap<I64Vec3, AbstractCluster>; LOD_LEVELS],
    p: ClusterPos,
) -> SmallVec<[ClusterPos; 8]> {
    // Higher lod
    if p.level < LOD_LEVELS - 1 {
        let size1 = cluster_size(p.level as u32 + 1) as i64;
        let pos1 = p.pos - p.pos.map(|v| v.rem_euclid(size1));

        if let Some(acluster) = aclusters[p.level + 1].get(&pos1) {
            return if rclusters[p.level + 1]
                .get(&pos1)
                .map_or(false, |v| v.mesh_available.load(atomic::Ordering::Relaxed))
                || acluster.occluded
                || (acluster.ready.load(atomic::Ordering::Relaxed)
                    && acluster.empty.load(atomic::Ordering::Relaxed))
            {
                smallvec![ClusterPos::new(p.level + 1, pos1)]
            } else {
                smallvec![]
            };
        }
    }

    if p.level == 0 {
        return smallvec![p];
    }

    let mut out = SmallVec::<[ClusterPos; 8]>::new();
    let size0 = cluster_size(p.level as u32 - 1) as i64;

    // Lower lod
    let mut state = true;
    let mut state_n = 0;

    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                let pos0 = p.pos + glm::vec3(x, y, z) * size0;

                if let Some(acluster) = aclusters[p.level - 1].get(&pos0) {
                    state_n += 1;

                    if rclusters[p.level - 1]
                        .get(&pos0)
                        .map_or(false, |v| v.mesh_available.load(atomic::Ordering::Relaxed))
                        || acluster.occluded
                        || (acluster.ready.load(atomic::Ordering::Relaxed)
                            && acluster.empty.load(atomic::Ordering::Relaxed))
                    {
                        out.push(ClusterPos::new(p.level - 1, pos0));
                    } else {
                        state = false;
                        break;
                    }
                }
            }
        }
    }

    if state {
        if state_n > 0 {
            out
        } else {
            smallvec![p]
        }
    } else {
        smallvec![]
    }
}

/// Returns `true` if cluster can be replaced by lower or higher lod clusters or if cluster's replacement is itself.
fn is_cluster_replacement_exists(layout: &[HashSet<I64Vec3>; LOD_LEVELS], p: ClusterPos) -> bool {
    // Higher lod
    if p.level < LOD_LEVELS - 1 {
        let size1 = cluster_size(p.level as u32 + 1) as i64;
        let lpos1 = (p.pos - p.pos.map(|v| v.rem_euclid(size1))) / size1;

        if layout[p.level + 1].contains(&lpos1) {
            return true;
        }
    }
    if p.level == 0 {
        return false;
    }

    // Lower lod
    let size0 = cluster_size(p.level as u32 - 1) as i64;
    let mut state = true;

    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                let lpos0 = p.pos / size0 + glm::vec3(x, y, z);

                if !layout[p.level - 1].contains(&lpos0) {
                    state = false;
                    break;
                }
            }
        }
    }
    if state {
        return true;
    }

    // Current lod
    let size = cluster_size(p.level as u32) as i64;
    let lpos = p.pos / size;

    layout[p.level].contains(&lpos)
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

        for i in 0..LOD_LEVELS {
            for (_, cl) in &mut self.aclusters[i] {
                cl.occlusion_outdated = true;
            }
        }
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

                let lock0 = cluster0.cluster.try_read();
                let lock1 = cluster1.cluster.try_write();

                if lock0.is_ok() && lock1.is_ok() {
                    let cluster = lock0.unwrap();
                    let mut side_cluster = lock1.unwrap();
                    let cluster = cluster.as_ref().unwrap();
                    let side_cluster = side_cluster.as_mut().unwrap();

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

    // TODO: make output positions in real units; don't forget to change that in usage of layout
    fn calc_cluster_layout(&self) -> [HashSet<I64Vec3>; LOD_LEVELS] {
        const R: i64 = (LOD0_RANGE / cluster::SIZE) as i64;
        const D: i64 = R * 2 + 1;

        let mut cluster_layout = vec![HashSet::with_capacity(512); LOD_LEVELS];
        let mut occupancy_masks = [[[DenseBitSet::new(); D as usize]; D as usize]; D as usize];
        let mut fill_masks = [[[DenseBitSet::new(); D as usize]; D as usize]; D as usize];

        fn min_dist(cpos: ClusterPos, spos: DVec3) -> f64 {
            let size = cluster_size(cpos.level as u32) as i64;
            let center: DVec3 = glm::convert(cpos.pos.add_scalar(size / 2));
            let halfsize = (size / 2) as f64;
            let xyz = spos - center;

            ((xyz.x.abs() - halfsize).max(0.0).powi(2)
                + (xyz.y.abs() - halfsize).max(0.0).powi(2)
                + (xyz.z.abs() - halfsize).max(0.0).powi(2))
            .sqrt()
        }

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
                        let dist = min_dist(ClusterPos::new(i, pos * cluster_size), self.stream_pos);

                        if dist <= r
                            && dist <= (self.render_distance as f64)
                            && !occupancy_masks[x as usize][y as usize][z as usize].get_bit(i)
                        {
                            let mut p = xyz;

                            // Propagate occupation
                            for j in (i + 1)..LOD_LEVELS {
                                let stream_pos_n = cluster_aligned_pos(self.stream_pos, j as u32 - 1);
                                let n = stream_pos_n.map(|v| v.rem_euclid(2));

                                p = (p.add_scalar(R) + n) / 2;
                                occupancy_masks[p[0] as usize][p[1] as usize][p[2] as usize].set_bit(j, true);

                                if j == i + 1 {
                                    fill_masks[p[0] as usize][p[1] as usize][p[2] as usize].set_bit(j, true);
                                }
                            }
                        }
                    }
                }
            }

            let stream_pos_i1 = cluster_aligned_pos(self.stream_pos, i as u32 + 1);

            // Calculate new cluster positions
            for x in 0..D {
                for y in 0..D {
                    for z in 0..D {
                        if !fill_masks[x as usize][y as usize][z as usize].get_bit(i + 1) {
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
                                        && (occupancy_masks[p[0]][p[1]][p[2]].get_bit(i))
                                    {
                                        continue;
                                    }

                                    let pos = pos + xyz2;

                                    // TODO: remove
                                    if cluster_layout[i].contains(&pos) {
                                        panic!("CONTAINS");
                                    }

                                    cluster_layout[i].insert(pos);
                                }
                            }
                        }
                    }
                }
            }
        }

        cluster_layout.try_into().unwrap()
    }

    pub fn update(&mut self, overworld: &mut Overworld) {
        let cluster_layout = self.calc_cluster_layout();

        // Add/remove clusters
        {
            let curr_time_secs = Instant::now().elapsed().as_secs();

            self.clusters_to_remove.clear();
            self.clusters_to_add.clear();

            for i in 0..LOD_LEVELS {
                // Remove unnecessary clusters
                let clusters = &self.clusters;
                let aclusters = &mut self.aclusters;
                let clusters_to_remove = &mut self.clusters_to_remove;

                // TODO: remove code duplicates
                overworld.loaded_clusters[i].retain(|pos, ocluster| {
                    // TODO: add this check
                    // if (curr_time_secs - ocluster.creation_time_secs) < 5
                    //     || cluster_layout[i].contains(&(pos / (cluster_size(i as u32) as i64)))

                    let lpos = pos / cluster_size(i as u32) as i64;
                    let inside_layout = cluster_layout[i].contains(&lpos);

                    if !inside_layout {
                        let p = ClusterPos::new(i, *pos);
                        clusters_to_remove.insert(p, is_cluster_replacement_exists(&cluster_layout, p));
                    }
                    inside_layout
                });

                // Account for non-removed clusters due to no replacement upon previous remove attempt
                for (pos, _) in &self.clusters[i] {
                    let lpos = pos / cluster_size(i as u32) as i64;

                    if !cluster_layout[i].contains(&lpos) {
                        let p = ClusterPos::new(i, *pos);
                        clusters_to_remove.insert(p, is_cluster_replacement_exists(&cluster_layout, p));
                    }
                }

                aclusters[i].retain(|pos, acluster| {
                    let lpos = pos / cluster_size(i as u32) as i64;
                    let occluded = acluster.occluded_new;
                    let inside_layout = cluster_layout[i].contains(&lpos);
                    let ready = acluster.ready.load(atomic::Ordering::Relaxed);
                    let empty = acluster.empty.load(atomic::Ordering::Relaxed);

                    // TODO: add this check
                    // if (curr_time_secs - ocluster.creation_time_secs) < 5
                    //     || cluster_layout[i].contains(&(pos / (cluster_size(i as u32) as i64)))

                    let optimized = occluded || (ready && empty);

                    if !inside_layout || optimized {
                        if overworld.loaded_clusters[i].remove(pos).is_some() {
                            let p = ClusterPos::new(i, *pos);
                            clusters_to_remove
                                .insert(p, !optimized && is_cluster_replacement_exists(&cluster_layout, p));
                        }
                    }
                    inside_layout
                });

                // Add missing clusters
                for lpos in &cluster_layout[i] {
                    let pos = lpos * cluster_size(i as u32) as i64;

                    if let hash_map::Entry::Vacant(entry) = self.aclusters[i].entry(pos) {
                        entry.insert(AbstractCluster {
                            occluded: false,
                            occluded_new: false,
                            occlusion_outdated: false,
                            empty: AtomicBool::new(true),
                            ready: AtomicBool::new(false),
                        });
                    }

                    let acluster = self.aclusters[i].get_mut(&pos).unwrap();
                    acluster.occluded = acluster.occluded_new;
                    acluster.occlusion_outdated = false;

                    if acluster.occluded
                        || (acluster.empty.load(atomic::Ordering::Relaxed)
                            && acluster.ready.load(atomic::Ordering::Relaxed))
                    {
                        continue;
                    }

                    if let hash_map::Entry::Vacant(entry) = overworld.loaded_clusters[i].entry(pos) {
                        entry.insert(Arc::new(OverworldCluster {
                            cluster: RwLock::new(None),
                            creation_time_secs: curr_time_secs,
                            generated: AtomicBool::new(false),
                            generating: AtomicBool::new(false),
                        }));

                        // Previous entity may exist if entry from `self.clusters`
                        // wasn't removed due to loading of lower/higher lod of clusters.
                        let prev_entity = self.clusters[i]
                            .get(&pos)
                            .map_or(scene::Entity::NULL, |v| v.entity);

                        self.clusters[i].insert(
                            pos,
                            RenderCluster {
                                entity: prev_entity,
                                available: AtomicBool::new(false),
                                changed: AtomicBool::new(false),
                                needs_occlusion_fill: AtomicBool::new(true),
                                mesh_changed: AtomicBool::new(false),
                                edge_occlusion: AtomicU8::new(0),
                                mesh_available: AtomicBool::new(false),
                            },
                        );
                        self.clusters_to_add.push(ClusterPos::new(i, pos));
                    }
                }
            }

            for i in 0..LOD_LEVELS {
                for lpos in &cluster_layout[i] {
                    let pos = lpos * cluster_size(i as u32) as i64;
                    let occluded = check_cluster_vis_occlusion(
                        &self.clusters,
                        &self.aclusters,
                        &cluster_layout,
                        i as u32,
                        pos,
                        self.stream_pos,
                    );
                    self.aclusters[i].get_mut(&pos).unwrap().occluded_new = occluded;
                }
            }
        }

        let sorted_layout: Vec<_> = cluster_layout
            .par_iter()
            .enumerate()
            .map(|(i, level)| {
                let cluster_size = cluster_size(i as u32) as i64;
                let mut clusters: Vec<_> = level
                    .iter()
                    .map(|pos| {
                        let pos = pos * cluster_size;
                        ClusterPosD {
                            pos,
                            distance: glm::distance(
                                &glm::convert(pos.add_scalar(cluster_size / 2)),
                                &self.stream_pos,
                            ),
                        }
                    })
                    .collect();
                clusters.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
                clusters
            })
            .collect();

        // Generate clusters
        {
            // TODO: utilize maximum amount of available cpu resources in this threadpool
            // TODO: reduce stalls (by waiting for some clusters generation?) before update()
            let max_clusters_in_process = rayon::current_num_threads() as u32;

            'l: for (i, level) in sorted_layout.iter().enumerate() {
                for lpos in level {
                    let ocluster = &overworld.loaded_clusters[i].get(&lpos.pos);
                    if ocluster.is_none() {
                        continue;
                    }
                    let ocluster = ocluster.unwrap();

                    let clusters_in_process = Arc::clone(&self.clusters_in_process);
                    let curr_clusters_in_process = clusters_in_process.load(atomic::Ordering::Acquire);

                    if curr_clusters_in_process >= max_clusters_in_process {
                        break 'l;
                    }

                    if !ocluster.generating.load(atomic::Ordering::Relaxed)
                        && !ocluster.generated.load(atomic::Ordering::Acquire)
                    {
                        let ocluster = Arc::clone(ocluster);
                        let pos = lpos.pos;
                        let node_size = 2_u32.pow(i as u32);
                        let main_registry = Arc::clone(&self.registry);
                        let device = Arc::clone(&self.device);

                        ocluster.generating.store(true, atomic::Ordering::Relaxed);
                        clusters_in_process.fetch_add(1, atomic::Ordering::Relaxed);
                        rayon::spawn(move || {
                            let mut cluster = cluster::new(main_registry.registry(), &device, node_size);
                            generator::generate_cluster(&mut cluster, &main_registry, pos);
                            *ocluster.cluster.write().unwrap() = Some(cluster);

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
            // Mark changes
            overworld
                .loaded_clusters
                .par_iter()
                .enumerate()
                .for_each(|(i, level)| {
                    level.par_iter().for_each(|(pos, ocluster)| {
                        let rcluster = &self.clusters[i][pos];

                        let available = !ocluster.generating.load(atomic::Ordering::Relaxed);
                        rcluster.available.store(available, atomic::Ordering::Relaxed);
                        if !available {
                            return;
                        }

                        let cluster_pos = ClusterPos::new(i, *pos);
                        let cluster = ocluster.cluster.read().unwrap();
                        if cluster.is_none() {
                            return;
                        }
                        let cluster = cluster.as_ref().unwrap();
                        let cluster_changed = cluster.changed();

                        if cluster_changed {
                            rcluster.changed.store(true, atomic::Ordering::Relaxed);
                            let side_clusters = self.find_side_clusters(overworld, cluster_pos);

                            for p in &side_clusters {
                                self.clusters[p.level][&p.pos]
                                    .needs_occlusion_fill
                                    .store(true, atomic::Ordering::Relaxed);
                            }

                            // Calculate full edge occlusion checks
                            let edge_check_may_update =
                                (rcluster.edge_occlusion.load(atomic::Ordering::Relaxed) >> 6 & 1) == 0;
                            if edge_check_may_update {
                                let mut edge_occlusion = 0b01000000; // set 'edge occlusion acquired' flag (7th bit)
                                for i in 0..6 {
                                    let occluded = cluster.check_edge_fully_occluded(Facing::from_u8(i));
                                    edge_occlusion |= (occluded as u8) << i;
                                }
                                let empty = cluster.is_empty();

                                rcluster
                                    .edge_occlusion
                                    .store(edge_occlusion, atomic::Ordering::Relaxed);
                                self.aclusters[i][pos]
                                    .empty
                                    .store(empty, atomic::Ordering::Relaxed);
                                self.aclusters[i][pos]
                                    .ready
                                    .store(true, atomic::Ordering::Relaxed);
                            }
                        }
                    });
                });

            // Collect changes
            for (i, level) in overworld.loaded_clusters.iter().enumerate() {
                for (pos, ocluster) in level {
                    let rcluster = &self.clusters[i][pos];
                    if !rcluster.available.load(atomic::Ordering::Relaxed)
                        || ocluster.cluster.read().unwrap().is_none()
                    {
                        continue;
                    }

                    let cluster_pos = ClusterPos::new(i, *pos);

                    if rcluster.needs_occlusion_fill.load(atomic::Ordering::Relaxed) {
                        let side_clusters = self.find_side_clusters(overworld, cluster_pos);
                        let mut ready_to_update = true;

                        for p in &side_clusters {
                            if !overworld.loaded_clusters[p.level][&p.pos]
                                .generated
                                .load(atomic::Ordering::Relaxed)
                            {
                                ready_to_update = false;
                                break;
                            }
                        }

                        if ready_to_update {
                            rcluster
                                .needs_occlusion_fill
                                .store(false, atomic::Ordering::Relaxed);

                            for p in side_clusters {
                                self.side_occlusion_work
                                    .sender
                                    .send(ClusterSidePair(p, cluster_pos))
                                    .unwrap();
                            }
                        }
                    }
                }
            }

            // Update cluster outer side occlusions
            // Note: parallelize avoiding deadlocks between side clusters
            let side_pair_count = self.side_occlusion_work.sender.len();
            self.side_occlusion_work
                .process_count
                .store(side_pair_count as u32, atomic::Ordering::Relaxed);
            (0..num_cpus::get()).into_par_iter().for_each(|_| {
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
                            && !rcluster.needs_occlusion_fill.load(atomic::Ordering::Relaxed)
                        {
                            let mut cluster = ocluster.cluster.write().unwrap();

                            cluster.as_mut().unwrap().update_mesh();
                            rcluster.mesh_changed.store(true, atomic::Ordering::Relaxed);
                            rcluster.changed.store(false, atomic::Ordering::Relaxed);

                            // Remove 'edge occlusion acquired' flag (7th bit)
                            rcluster
                                .edge_occlusion
                                .fetch_and(0b00111111, atomic::Ordering::Relaxed);
                        }
                    });
                });
        }

        let c = overworld.loaded_clusters.iter().fold(0, |i, v| {
            i + v.iter().fold(0, |i, v| {
                i + if let Some(cl) = v.1.cluster.read().unwrap().as_ref() {
                    (cl.vertex_mesh().vertex_count() > 0) as u32
                } else {
                    0
                }
            })
        });
        println!("L {}", c);
    }

    pub fn update_renderer(&mut self, overworld: &mut Overworld) {
        let renderer = self.renderer.lock().unwrap();
        let scene = renderer.scene();

        let c = self.clusters.iter().fold(0, |i, v| i + v.len());
        let ent_c = renderer.scene().entities().read().unwrap().len();

        println!("R {}, entities {}", c, ent_c);

        let aclusters = &self.aclusters;
        let rclusters = &mut self.clusters;

        self.clusters_to_remove.retain(|k, &mut v| {
            let repl = cluster_replacement(rclusters, aclusters, *k);

            !v || (repl.iter().all(|p| {
                if let Some(cl) = rclusters[p.level].get(&p.pos) {
                    !renderer.is_vertex_mesh_updating(cl.entity)
                } else {
                    true
                }
            }) && !repl.is_empty())
        });

        component::remove_entities(
            scene,
            &self
                .clusters_to_remove
                .iter()
                .map(|(v, _)| rclusters[v.level][&v.pos].entity)
                .collect::<Vec<_>>(),
        );
        for (v, _) in &self.clusters_to_remove {
            rclusters[v.level].remove(&v.pos);
        }

        // Enable visibility of loaded clusters
        // Note: must be called between 'removing clusters' and 'creating new ones'.
        {
            let renderer_comps = scene.storage::<component::Renderer>();
            let mut renderer_comps = renderer_comps.write();

            for (i, level) in rclusters.iter().enumerate() {
                for (p, rcluster) in level {
                    if rcluster.entity == scene::Entity::NULL {
                        continue;
                    }
                    if renderer_comps.get_mut(rcluster.entity).unwrap().visible() {
                        continue;
                    }

                    let size1 = cluster_size(i as u32 + 1) as i64;
                    let pos1 = p - p.map(|v| v.rem_euclid(size1));
                    let mut state = true;

                    if i + 1 < LOD_LEVELS {
                        if let Some(rcl) = rclusters[i + 1].get(&pos1) {
                            if renderer_comps.get_mut(rcl.entity).map_or(false, |v| v.visible()) {
                                let repl =
                                    cluster_replacement(rclusters, aclusters, ClusterPos::new(i + 1, pos1));

                                state = repl.iter().all(|p| {
                                    if let Some(cl) = rclusters[p.level].get(&p.pos) {
                                        !renderer.is_vertex_mesh_updating(cl.entity)
                                    } else {
                                        true
                                    }
                                }) && !repl.is_empty();
                            }
                        }
                    }
                    if state {
                        renderer_comps.get_mut(rcluster.entity).unwrap().set_visible(true);
                    }
                }
            }
        }

        // Reserve entities from scene
        {
            let mut entities = scene.entities().write().unwrap();

            for p in &self.clusters_to_add {
                let rcluster = rclusters[p.level].get_mut(&p.pos).unwrap();

                if rcluster.entity == scene::Entity::NULL {
                    rcluster.entity = entities.create();
                }
            }
        }

        let transform_comps = scene.storage::<component::Transform>();
        let renderer_comps = scene.storage::<component::Renderer>();
        let vertex_mesh_comps = scene.storage::<component::VertexMesh>();
        let mut transform_comps = transform_comps.write();
        let mut renderer_comps = renderer_comps.write();
        let mut vertex_mesh_comps = vertex_mesh_comps.write();

        // Set components
        for p in &self.clusters_to_add {
            let pos = &p.pos;
            let transform_comp = component::Transform::new(
                Vec3::new(pos.x as f32, pos.y as f32, pos.z as f32),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let mut renderer_comp = component::Renderer::new(&renderer, self.cluster_mat_pipeline, false);
            renderer_comp.set_visible(false);

            let entity = rclusters[p.level][&p.pos].entity;
            transform_comps.set(entity, transform_comp);
            renderer_comps.set(entity, renderer_comp);
        }

        // Update meshes
        for (i, level) in overworld.loaded_clusters.iter().enumerate() {
            for (pos, ocluster) in level {
                let rcluster = &rclusters[i][pos];
                if rcluster.mesh_changed.swap(false, atomic::Ordering::Relaxed) {
                    let cluster = ocluster.cluster.read().unwrap();
                    let mesh = cluster.as_ref().unwrap().vertex_mesh();
                    vertex_mesh_comps.set(rcluster.entity, component::VertexMesh::new(&mesh.raw()));
                    rcluster.mesh_available.store(true, atomic::Ordering::Relaxed);
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
        cluster_mat_pipeline,
        render_distance: 0,
        stream_pos: DVec3::new(0.0, 0.0, 0.0),
        clusters: Default::default(),
        aclusters: Default::default(),
        side_occlusion_work: SideOcclusionWorkSync {
            sender: occ_s,
            receiver: occ_r,
            process_count: Default::default(),
        },
        clusters_to_remove: Default::default(),
        clusters_to_add: vec![],
        clusters_in_process: Arc::new(AtomicU32::new(0)),
    }
}
