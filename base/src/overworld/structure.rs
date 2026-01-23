pub mod world;

use crate::overworld::facing::Facing;
use crate::overworld::generator::{OverworldGenerator, StructureCache};
use crate::overworld::position::BlockPos;
use crate::overworld::raw_cluster::RawCluster;
use bit_vec::BitVec;
use common::glm;
use common::glm::{I64Vec3, U64Vec3};
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

/// Returns whether the structure can be generated at specified center position.
pub type GenPosCheckFn =
    fn(structure: &Structure, generator: &OverworldGenerator, center_pos: BlockPos) -> bool;

/// Fills the specified cluster with structure's blocks.
/// `cluster_origin` specifies the relative (to the structure) position of the (0,0,0) block of the cluster.
pub type GenFn = fn(
    structure: &Structure,
    generator: &OverworldGenerator,
    structure_seed: u64,
    cluster_origin: BlockPos,
    cluster: &mut RawCluster,
    structure_state: Arc<OnceLock<Box<dyn StructureCache>>>,
);

/// Returns potential spawn point.
pub type GenSpawnPointFn = fn(
    structure: &Structure,
    generator: &OverworldGenerator,
    structure_seed: u64,
    structure_state: Arc<OnceLock<Box<dyn StructureCache>>>,
) -> BlockPos;

pub struct Structure {
    uid: u32,
    /// Maximum size in blocks
    max_size: U64Vec3,
    /// Min-max spacing range in blocks between structures of this type.
    /// Octants are tightly packed based on this range and the structure size.
    /// This also introduces randomness to generated center positions of structures.
    /// A very specific range (450..450) has no randomness because the spacing.
    spacing_range: Range<u64>,
    gen_pos_check_fn: GenPosCheckFn,
    gen_fn: GenFn,
    gen_spawn_point_fn: Option<GenSpawnPointFn>,
}

impl Structure {
    /// `uid` must be unique to avoid generating different structures at the same cluster
    pub fn new(
        uid: u32,
        max_size: U64Vec3,
        spacing_range: Range<u64>,
        gen_pos_check_fn: GenPosCheckFn,
        gen_fn: GenFn,
        gen_spawn_point_fn: Option<GenSpawnPointFn>,
    ) -> Structure {
        Structure {
            uid,
            max_size,
            spacing_range,
            gen_pos_check_fn,
            gen_fn,
            gen_spawn_point_fn,
        }
    }

    pub fn uid(&self) -> u32 {
        self.uid
    }

    pub fn max_size(&self) -> U64Vec3 {
        self.max_size
    }

    /// The structures of this type are spaced at least this far apart.
    pub fn min_spacing(&self) -> u64 {
        self.spacing_range.start
    }

    /// The structures of this type are spaced at most this far apart.
    pub fn max_spacing(&self) -> u64 {
        self.spacing_range.end
    }

    /// Octant size is the side-length of the area of generation of a single structure.
    pub fn octant_size(&self) -> u64 {
        self.max_size.max() + self.max_spacing() / 2
    }

    pub fn calc_random_range(&self) -> Range<u64> {
        let gen_padding = (self.max_size.max() / 2).max(self.min_spacing() / 2);
        gen_padding..(self.octant_size() - gen_padding)
    }

    pub fn check_gen_pos(&self, generator: &OverworldGenerator, center_pos: BlockPos) -> bool {
        (self.gen_pos_check_fn)(self, generator, center_pos)
    }

    pub fn gen_cluster(
        &self,
        generator: &OverworldGenerator,
        structure_seed: u64,
        cluster_origin: BlockPos,
        cluster: &mut RawCluster,
        structure_cache: Arc<OnceLock<Box<dyn StructureCache>>>,
    ) {
        (self.gen_fn)(
            self,
            generator,
            structure_seed,
            cluster_origin,
            cluster,
            structure_cache,
        );
    }

    pub fn gen_spawn_point(
        &self,
        generator: &OverworldGenerator,
        structure_seed: u64,
        structure_cache: Arc<OnceLock<Box<dyn StructureCache>>>,
    ) -> Option<BlockPos> {
        self.gen_spawn_point_fn
            .map(|f| f(self, generator, structure_seed, structure_cache))
    }
}

pub struct StructuresIter<'a> {
    pub(super) generator: &'a OverworldGenerator,
    pub(super) structure: &'a Structure,
    pub(super) start_octant: I64Vec3,
    pub(super) max_search_radius: u32,
    pub(super) queue: VecDeque<I64Vec3>,
    pub(super) traversed_nodes: BitVec,
}

impl Iterator for StructuresIter<'_> {
    type Item = BlockPos;

    fn next(&mut self) -> Option<Self::Item> {
        let diam = self.max_search_radius as i64 * 2 - 1;
        let octant_size = self.structure.octant_size() as i64;

        let front_octant = self.queue.front().unwrap();
        {
            let rel_pos = (front_octant - self.start_octant).add_scalar(diam / 2);
            let idx_1d = (rel_pos.x * diam * diam + rel_pos.y * diam + rel_pos.x) as usize;

            if !self.traversed_nodes.get(idx_1d).unwrap() {
                self.traversed_nodes.set(idx_1d, true);

                let st_pos = self
                    .generator
                    .gen_structure_pos(self.structure, BlockPos(front_octant * octant_size));

                if st_pos.present {
                    return Some(st_pos.center_pos);
                }
            }
        }

        // Use breadth-first search
        while let Some(curr_octant) = self.queue.pop_front() {
            for dir in &Facing::DIRECTIONS {
                let dir: I64Vec3 = glm::convert(*dir);
                let next_pos = curr_octant + dir;

                let rel_pos = (next_pos - self.start_octant).add_scalar(diam / 2);
                if rel_pos.x < 0
                    || rel_pos.y < 0
                    || rel_pos.z < 0
                    || rel_pos.x >= diam
                    || rel_pos.y >= diam
                    || rel_pos.z >= diam
                {
                    continue;
                }

                let idx_1d = (rel_pos.z * diam * diam + rel_pos.y * diam + rel_pos.x) as usize;
                if self.traversed_nodes.get(idx_1d).unwrap() {
                    continue;
                }

                self.queue.push_back(next_pos);
                self.traversed_nodes.set(idx_1d, true);

                let st_pos = self
                    .generator
                    .gen_structure_pos(self.structure, BlockPos(next_pos * octant_size));

                if st_pos.present {
                    return Some(st_pos.center_pos);
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.max_search_radius.pow(3) as usize))
    }
}
