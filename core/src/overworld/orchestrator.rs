use std::hash::Hash;
use std::mem;
use std::sync::atomic::{AtomicBool, AtomicU8};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel as cb;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I32Vec3, I64Vec3};
use parking_lot::{Mutex, RwLockWriteGuard};
use rayon::prelude::*;
use smallvec::SmallVec;

use crate::overworld::cluster_part_set::ClusterPartSet;
use crate::overworld::facing::Facing;
use crate::overworld::generator::OverworldGenerator;
use crate::overworld::occluder::Occluder;
use crate::overworld::position::{BlockPos, ClusterPos};
use crate::overworld::raw_cluster::{CellInfo, RawCluster};
use crate::overworld::{
    raw_cluster, ClusterState, LoadedClusters, Overworld, OverworldCluster, TrackingCluster,
};
use crate::utils::{HashMap, HashSet, MO_RELAXED, MO_RELEASE};

pub const FORCED_LOAD_RANGE: usize = 128;

/// Manages overworld: adds/removes new clusters, loads them in parallel, etc.
pub struct OverworldOrchestrator {
    stream_pos: DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    loaded_clusters: LoadedClusters,
    overworld_generator: Arc<OverworldGenerator>,
    r_clusters: HashMap<ClusterPos, RCluster>,
}

struct RCluster {
    creation_time: Instant,
    /// Whether this cluster is invisible due to full occlusion by the neighbouring clusters
    occluded: AtomicBool,
    empty: Arc<AtomicBool>,
    /// A mask of 3x3 'sides' which are needed to be filled with intrinsics of neighbour clusters
    needs_auxiliary_fill_at: ClusterPartSet,
    /// Whether a particular Facing of this cluster can occlude a whole side of any other cluster
    side_occlusion: Arc<AtomicU8>,
}

impl RCluster {
    fn is_empty_or_occluded(&self) -> bool {
        self.empty.load(MO_RELAXED) || self.occluded.load(MO_RELAXED)
    }
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
    let cr = (xz_render_distance / RawCluster::SIZE as u64 / 2) as i64;
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

fn get_side_cluster_by_facing(pos: &ClusterPos, facing: Facing) -> ClusterPos {
    pos.offset(&glm::convert(*facing.direction()))
}

fn is_cluster_in_forced_load_range(pos: &ClusterPos, stream_pos: &DVec3) -> bool {
    let center_pos: DVec3 = glm::convert(pos.get().add_scalar(RawCluster::SIZE as i64 / 2));
    let distance = glm::distance(&center_pos, &stream_pos);
    distance <= FORCED_LOAD_RANGE as f64
}

/// Checks if cluster at `pos` is occluded at all sides by the neighbour clusters
fn is_cluster_visibly_occluded(
    r_clusters: &HashMap<ClusterPos, RCluster>,
    o_clusters: &HashMap<ClusterPos, OverworldCluster>,
    pos: &ClusterPos,
) -> bool {
    for i in 0..6 {
        let facing = Facing::from_u8(i);
        let neighbour_pos = get_side_cluster_by_facing(pos, facing);

        if let Some(ocl) = o_clusters.get(&neighbour_pos) {
            if ocl.dirty.load(MO_RELAXED) {
                // Dirty neighbours do not have updated RCluster::side_occlusion (used below).
                return false;
            }
        }

        if let Some(rcl) = r_clusters.get(&neighbour_pos) {
            let side_occlusion = rcl.side_occlusion.load(MO_RELAXED);
            let side_occluded = ((side_occlusion >> (facing.mirror() as u8)) & 1) == 1;

            if !side_occluded {
                return false;
            }
        } else {
            return false;
        }
    }

    true
}

struct ClustersDiff {
    sorted_layout: Vec<ClusterPosDistance>,
    new: Vec<ClusterPos>,
    to_remove: Vec<ClusterPos>,
}

fn get_clusters_difference(
    stream_pos: &DVec3,
    xz_render_distance: u64,
    y_render_distance: u64,
    r_clusters: &HashMap<ClusterPos, RCluster>,
) -> ClustersDiff {
    let curr_t = Instant::now();
    let layout = calc_cluster_layout(stream_pos, xz_render_distance, y_render_distance);

    let mut sorted_layout: Vec<_> = layout
        .iter()
        .map(|p| ClusterPosDistance {
            pos: *p,
            distance: glm::distance2(
                &glm::convert::<_, DVec3>(p.get().add_scalar(RawCluster::SIZE as i64 / 2)),
                &stream_pos,
            ),
        })
        .collect();
    // Sort by lowest distance to streaming position
    sorted_layout.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));

    let mut res = ClustersDiff {
        sorted_layout,
        new: Vec::with_capacity(8192),
        to_remove: Vec::with_capacity(8192),
    };

    // Collect new clusters
    for d_pos in &res.sorted_layout {
        if !r_clusters.contains_key(&d_pos.pos) {
            res.new.push(d_pos.pos);
        }
    }

    // Collect clusters to remove
    for (pos, r_cluster) in r_clusters {
        let in_layout = layout.contains(pos);
        let timeout = (curr_t - r_cluster.creation_time).as_secs() >= 3;

        if !in_layout && timeout {
            res.to_remove.push(*pos);
            continue;
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
                creation_time: Instant::now(),
                occluded: AtomicBool::new(false),
                empty: Arc::new(AtomicBool::new(false)),
                needs_auxiliary_fill_at: ClusterPartSet::ALL,
                side_occlusion: Arc::new(Default::default()),
            },
        );
        o_clusters.insert(*pos, OverworldCluster::new());
    }
}

fn set_auxiliary_fill_need_for_neighbours(pos: &ClusterPos, r_clusters: &mut HashMap<ClusterPos, RCluster>) {
    for neighbour in get_side_clusters(pos) {
        let Some(r_cluster) = r_clusters.get_mut(&neighbour) else {
            continue;
        };
        r_cluster
            .needs_auxiliary_fill_at
            .set_from_relation(&neighbour, pos);
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
        set_auxiliary_fill_need_for_neighbours(pos, r_clusters);
    }
}

fn offload_clusters(
    to_offload: &[(ClusterPos, ClusterState)],
    r_clusters: &mut HashMap<ClusterPos, RCluster>,
    o_clusters: &HashMap<ClusterPos, OverworldCluster>,
) {
    for (pos, offload_state) in to_offload {
        let r_cluster = r_clusters.get_mut(pos).unwrap();
        let o_cluster = o_clusters.get(pos).unwrap();

        o_cluster.state.store(*offload_state as u32, MO_RELAXED);
        *o_cluster.cluster.write() = None;
        r_cluster.needs_auxiliary_fill_at = ClusterPartSet::NONE;

        set_auxiliary_fill_need_for_neighbours(pos, r_clusters);
    }
}

#[allow(unreachable_code)] // TODO: remove after IDE fix of break in let-else statement
fn auxiliary_merge_worker(
    channel: (cb::Sender<IntrinsicsUpdate>, cb::Receiver<IntrinsicsUpdate>),
    o_clusters: &HashMap<ClusterPos, OverworldCluster>,
) {
    loop {
        let Ok(update) = channel.1.try_recv() else {
            break;
        };

        let a_ocl = o_clusters.get(&update.a).unwrap();
        let Some(mut a_tcl) = a_ocl.cluster.try_write() else {
            channel.0.send(update).unwrap();
            continue;
        };
        let a_tcl = a_tcl.as_mut().unwrap();

        let b_data_source = if let Some(neighbour) = o_clusters.get(&update.b) {
            match neighbour.state() {
                ClusterState::Loaded => IntrinsicSource::Cluster,
                ClusterState::OffloadedEmpty => IntrinsicSource::Value(Occluder::EMPTY),
                ClusterState::OffloadedOccluded => IntrinsicSource::Value(Occluder::FULL),
                _ => unreachable!(),
            }
        } else {
            IntrinsicSource::Value(Occluder::EMPTY)
        };
        let b_offset: I32Vec3 = glm::convert(update.b.get() - update.a.get());

        if let IntrinsicSource::Value(occluder) = b_data_source {
            // There is no cluster at position `b` => fill `a` with empty auxiliary cells.
            a_tcl.raw.clear_auxiliary_cells(
                b_offset,
                CellInfo {
                    entity_id: Default::default(),
                    block_id: 0,
                    occluder,
                    light_level: Default::default(),
                    liquid_state: Default::default(),
                    active: false,
                },
            );
            continue;
        }

        let b_ocl = o_clusters.get(&update.b).unwrap();
        let Some(mut b_tcl) = b_ocl.cluster.try_write() else {
            channel.0.send(update).unwrap();
            continue;
        };
        let b_tcl = b_tcl.as_mut().unwrap();

        // Mutually update `cluster` and `neighbour_cluster` intrinsics.
        // First, propagate lighting in both clusters,
        // and only then update intrinsics (which are also related by lighting).
        let dirty_parts = a_tcl.raw.propagate_outer_lighting(&b_tcl.raw, b_offset);
        a_tcl.dirty_parts |= dirty_parts;
        let dirty_parts = b_tcl.raw.propagate_outer_lighting(&a_tcl.raw, -b_offset);
        b_tcl.dirty_parts |= dirty_parts;

        // Mutually paste intrinsics.
        a_tcl.raw.paste_auxiliary_cells(&b_tcl.raw, b_offset);
        b_tcl.raw.paste_auxiliary_cells(&a_tcl.raw, -b_offset);

        // If either cluster has been changed, mark OverworldCluster as changed
        // to handle it in further updates of OverworldStreamer.
        if a_tcl.dirty_parts.has_any() {
            a_ocl.dirty.store(true, MO_RELAXED);
        }
        if b_tcl.dirty_parts.has_any() {
            b_ocl.dirty.store(true, MO_RELAXED);
        }
    }
}

#[derive(Eq, PartialEq)]
enum IntrinsicSource {
    Cluster,
    Value(Occluder),
}

#[derive(Eq, PartialEq, Hash)]
struct IntrinsicsUpdate {
    /// An OverworldCluster for this position MUST be present.
    a: ClusterPos,
    /// An OverworldCluster for this position MAY be present.
    b: ClusterPos,
}

pub struct OverworldUpdateResult {
    /// Clusters that have changed since the last update
    pub processed_dirty_clusters: HashMap<ClusterPos, ClusterPartSet>,
    /// When a cluster C has dirty parts, respective auxiliary parts of neighbour clusters
    /// must be updated with cells from opposite parts of C.  
    /// This represents which clusters have their auxiliary parts updated.
    pub updated_auxiliary_parts: Vec<(ClusterPos, ClusterPartSet)>,
    pub new_clusters: Vec<ClusterPos>,
    pub removed_clusters: Vec<ClusterPos>,
    pub offloaded_clusters: Vec<ClusterPos>,
}

impl OverworldOrchestrator {
    const MIN_XZ_RENDER_DISTANCE: u64 = 128;
    const MAX_XZ_RENDER_DISTANCE: u64 = 1024;
    const MIN_Y_RENDER_DISTANCE: u64 = 128;
    const MAX_Y_RENDER_DISTANCE: u64 = 512;

    pub fn new(overworld: &Overworld) -> Self {
        Self {
            stream_pos: Default::default(),
            xz_render_distance: 128,
            y_render_distance: 128,
            loaded_clusters: Arc::clone(overworld.loaded_clusters()),
            overworld_generator: Arc::clone(overworld.generator()),
            r_clusters: Default::default(),
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

    /// Returns affected neighbour clusters
    fn collect_dirty_affected_clusters<F: FnMut(ClusterPos)>(
        current_pos: &ClusterPos,
        dirty_parts: ClusterPartSet,
        mut output: F,
    ) {
        for dirty_side_idx in dirty_parts.iter_ones() {
            let changed_dir: I64Vec3 = glm::convert(raw_cluster::neighbour_index_to_dir(dirty_side_idx));

            let p0 = current_pos.get() / (RawCluster::SIZE as i64);
            let p1 = p0 + changed_dir;

            let min = p0.inf(&p1);
            let max = p0.sup(&p1);

            for x in min.x..=max.x {
                for y in min.y..=max.y {
                    for z in min.z..=max.z {
                        let neighbour_pos = ClusterPos::new(glm::vec3(x, y, z) * (RawCluster::SIZE as i64));
                        if &neighbour_pos == current_pos {
                            continue;
                        }
                        output(neighbour_pos);
                    }
                }
            }
        }
    }

    pub fn collect_dirty_clusters(&self) -> Vec<ClusterPos> {
        let o_clusters = self.loaded_clusters.write();
        let mut dirty_clusters = Vec::with_capacity(o_clusters.len());

        for (pos, o_cluster) in &*o_clusters {
            if !o_cluster.dirty.load(MO_RELAXED) {
                continue;
            }
            dirty_clusters.push(*pos);
        }

        dirty_clusters
    }

    /// Generates new clusters and their content. Updates and optimizes overworld cluster layout.
    pub fn update(&mut self, max_processing_time: Duration) -> OverworldUpdateResult {
        let t_start = Instant::now();
        let mut res = OverworldUpdateResult {
            processed_dirty_clusters: HashMap::with_capacity(1024),
            updated_auxiliary_parts: Vec::with_capacity(1024),
            new_clusters: Vec::with_capacity(1024),
            removed_clusters: Vec::with_capacity(1024),
            offloaded_clusters: Vec::with_capacity(1024),
        };
        // Lock loaded_clusters so no accessor can modify clusters
        let mut o_clusters = self.loaded_clusters.write();
        let r_clusters = &mut self.r_clusters;

        let clusters_diff = get_clusters_difference(
            &self.stream_pos,
            self.xz_render_distance,
            self.y_render_distance,
            &r_clusters,
        );

        // 1. Apply clusters difference
        remove_clusters(&clusters_diff.to_remove, r_clusters, &mut o_clusters);
        add_new_clusters(&clusters_diff.new, r_clusters, &mut o_clusters);

        // Write access is not needed anymore
        let o_clusters = RwLockWriteGuard::downgrade(o_clusters);

        // ------------------------------------------------------------------------------------------------------

        let mut dirty_clusters = HashMap::with_capacity(1024);

        // 2. Collect dirty clusters
        for (pos, o_cluster) in &*o_clusters {
            if !o_cluster.dirty.load(MO_RELAXED) {
                continue;
            }
            o_cluster.dirty.store(false, MO_RELAXED);

            let dirty_parts = {
                let mut cluster = o_cluster.cluster.write();
                mem::replace(&mut cluster.as_mut().unwrap().dirty_parts, ClusterPartSet::NONE)
            };
            dirty_clusters.insert(*pos, dirty_parts);
        }

        let to_offload = Mutex::new(Vec::with_capacity(dirty_clusters.len()));

        // 3. Check dirty clusters for emptiness, active blocks, and side occlusion
        // Note: this must be done for all dirty clusters in one orchestrator update
        dirty_clusters.par_iter().for_each(|(pos, _)| {
            let o_cluster = o_clusters.get(pos).unwrap();
            let t_cluster_guard = o_cluster.cluster.read();
            let t_cluster = t_cluster_guard.as_ref().unwrap();
            let r_cluster = r_clusters.get(pos).unwrap();

            let is_empty = t_cluster.is_empty();
            let has_active_blocks = t_cluster.has_active_blocks();

            let side_occlusion = (0..6).fold(0_u8, |prev, i| {
                let occluded = t_cluster.raw.is_side_fully_occluded(Facing::from_u8(i));
                prev | ((occluded as u8) << i)
            });

            r_cluster.side_occlusion.store(side_occlusion, MO_RELAXED);
            r_cluster.empty.store(is_empty, MO_RELAXED);
            o_cluster.has_active_blocks.store(has_active_blocks, MO_RELAXED);

            if is_empty {
                // Offload
                to_offload.lock().push((*pos, ClusterState::OffloadedEmpty));
            }
        });

        let mut to_offload = to_offload.into_inner();

        // 4. Check clusters for visibility occlusion
        // TODO: optimize this to iterate relevant clusters only
        for (pos, rcl) in &*r_clusters {
            let ocl = o_clusters.get(pos).unwrap();

            let is_occluded = is_cluster_visibly_occluded(&r_clusters, &o_clusters, &pos);
            rcl.occluded.store(is_occluded, MO_RELAXED);

            if is_occluded && !ocl.state().is_offloaded() {
                to_offload.push((*pos, ClusterState::OffloadedOccluded));
            }
        }

        // 5. Offload relevant clusters
        // Do not offload clusters close to streaming position
        to_offload = to_offload
            .into_iter()
            .filter(|(pos, _)| !is_cluster_in_forced_load_range(pos, &self.stream_pos))
            .collect();
        // Offloaded clusters can't be dirty
        for (pos, _) in &to_offload {
            dirty_clusters.remove(pos);
        }
        offload_clusters(&to_offload, r_clusters, &*o_clusters);

        // 6. Mark respective auxiliary parts of affected clusters as dirty to fill them later
        for (curr_pos, dirty_parts) in &dirty_clusters {
            Self::collect_dirty_affected_clusters(curr_pos, *dirty_parts, |neighbour_pos| {
                let Some(rcl) = r_clusters.get_mut(&neighbour_pos) else {
                    return;
                };
                rcl.needs_auxiliary_fill_at
                    .set_from_relation(&neighbour_pos, curr_pos);
            });
        }

        let mut pairs_to_merge_sides = HashSet::<IntrinsicsUpdate>::with_capacity(1024);

        // 7. Collect cluster pairs to mutually update their auxiliary parts
        for (curr_pos, rcl) in &mut *r_clusters {
            if !rcl.needs_auxiliary_fill_at.has_any() {
                continue;
            }
            let fill_parts = rcl.needs_auxiliary_fill_at;
            let curr_ocl = o_clusters.get(curr_pos).unwrap();

            if !curr_ocl.state().is_writable() {
                continue;
            }

            let mut parts_to_update = ClusterPartSet::NONE;

            for part_idx in fill_parts.iter_ones() {
                let rel_dir: I64Vec3 = glm::convert(raw_cluster::neighbour_index_to_dir(part_idx));
                let neighbour_pos = curr_pos.offset(&rel_dir);

                if let Some(neighbour_ocl) = o_clusters.get(&neighbour_pos) {
                    if !neighbour_ocl.state().is_loaded() {
                        rcl.needs_auxiliary_fill_at.clear_from_idx(part_idx);
                        continue;
                    }
                }

                let update = IntrinsicsUpdate {
                    a: *curr_pos,
                    b: neighbour_pos,
                };
                let opposite_update = IntrinsicsUpdate {
                    a: neighbour_pos,
                    b: *curr_pos,
                };

                if !pairs_to_merge_sides.contains(&opposite_update) {
                    pairs_to_merge_sides.insert(update);
                }
                parts_to_update.set_from_idx(part_idx);
            }

            res.updated_auxiliary_parts.push((*curr_pos, parts_to_update));
        }

        let pair_merge_channel = cb::unbounded::<IntrinsicsUpdate>();

        // Unmark to-do merges and collect unique cluster positions
        for pair in pairs_to_merge_sides {
            if let Some(a) = r_clusters.get_mut(&pair.a) {
                a.needs_auxiliary_fill_at.clear_from_relation(&pair.a, &pair.b);
            }
            if let Some(b) = r_clusters.get_mut(&pair.b) {
                b.needs_auxiliary_fill_at.clear_from_relation(&pair.b, &pair.a);
            }
            pair_merge_channel.0.send(pair).unwrap();
        }

        // 8. Fill auxiliary cluster parts
        rayon::scope(|s| {
            for _ in 0..rayon::current_num_threads() {
                let channel = pair_merge_channel.clone();
                s.spawn(|_| {
                    auxiliary_merge_worker(channel, &o_clusters);
                })
            }
        });

        let mut clusters_generating = 0;
        let mut reloaded_clusters = Vec::with_capacity(128);

        // 9. Load(generate) clusters which are !ready or (not present in o_clusters and !RCluster::occluded)
        rayon::scope(|s| {
            let registry = self.overworld_generator.main_registry().registry();

            for d_pos in &clusters_diff.sorted_layout {
                let pos = &d_pos.pos;
                let o_cluster = o_clusters.get(pos).unwrap();
                let r_cluster = r_clusters.get(pos).unwrap();
                let state = o_cluster.state();

                let in_forced_load_range = is_cluster_in_forced_load_range(pos, &self.stream_pos);

                if state == ClusterState::Loaded
                    || (!in_forced_load_range && r_cluster.is_empty_or_occluded())
                {
                    continue;
                }
                if state.is_empty_or_occluded() {
                    reloaded_clusters.push(*pos);
                }

                s.spawn(|_| {
                    let mut cluster = self.overworld_generator.create_cluster();
                    self.overworld_generator.generate_cluster(&mut cluster, *pos);

                    *o_cluster.cluster.write() = Some(TrackingCluster::new(registry, cluster));
                    o_cluster.state.store(ClusterState::Loaded as u32, MO_RELEASE);
                    o_cluster.dirty.store(true, MO_RELEASE);
                });

                clusters_generating += 1;
                dirty_clusters.insert(*pos, ClusterPartSet::ALL);

                let t1 = Instant::now();
                if clusters_generating > 0 && t1.duration_since(t_start) >= max_processing_time {
                    break;
                }
            }
        });

        res.processed_dirty_clusters = dirty_clusters;
        res.new_clusters = clusters_diff.new;
        res.new_clusters.extend(reloaded_clusters);
        res.removed_clusters = clusters_diff.to_remove;
        res.offloaded_clusters.extend(to_offload.iter().map(|v| v.0));

        res
    }
}
