pub mod world;

use crate::game::overworld::block_component::Facing;
use crate::game::overworld::cluster::Cluster;
use crate::game::overworld::generator::{OverworldGenerator, StructureCache};
use crate::game::overworld::{cluster, Overworld};
use bit_vec::BitVec;
use engine::utils::UInt;
use nalgebra_glm as glm;
use nalgebra_glm::{I64Vec3, U64Vec3};
use once_cell::sync::OnceCell;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

/// Returns whether the structure can be generated at specified center position.
pub type GenPosCheckFn =
    fn(structure: &Structure, generator: &OverworldGenerator, center_pos: I64Vec3) -> bool;

/// Fills the specified cluster with structure's blocks
pub type GenFn = fn(
    structure: &Structure,
    generator: &OverworldGenerator,
    structure_seed: u64,
    center_pos: I64Vec3,
    cluster: &mut Cluster,
    structure_cache: Arc<OnceCell<Box<dyn StructureCache>>>,
);

pub struct Structure {
    uid: u32,
    /// Maximum size in blocks
    max_size: U64Vec3,
    /// Minimum spacing in clusters between structures of this type
    min_spacing: u64,
    /// Average spacing in clusters between structures of this type
    avg_spacing: u64,
    gen_pos_check_fn: GenPosCheckFn,
    gen_fn: GenFn,
}

impl Structure {
    /// `uid` must be unique to avoid generating different structures at the same cluster
    pub fn new(
        uid: u32,
        max_size: U64Vec3,
        min_spacing: u64,
        avg_spacing: u64,
        gen_pos_check_fn: GenPosCheckFn,
        gen_fn: GenFn,
    ) -> Structure {
        assert!(avg_spacing >= min_spacing);

        let size_in_clusters = UInt::div_ceil(max_size.max(), cluster::SIZE as u64);
        assert!(min_spacing >= size_in_clusters);

        Structure {
            uid,
            max_size,
            min_spacing,
            avg_spacing,
            gen_pos_check_fn,
            gen_fn,
        }
    }

    pub fn uid(&self) -> u32 {
        self.uid
    }

    pub fn max_size(&self) -> U64Vec3 {
        self.max_size
    }

    pub fn min_spacing(&self) -> u64 {
        self.min_spacing
    }

    pub fn avg_spacing(&self) -> u64 {
        self.avg_spacing
    }

    pub fn check_gen_pos(&self, generator: &OverworldGenerator, center_pos: I64Vec3) -> bool {
        (self.gen_pos_check_fn)(self, generator, center_pos)
    }

    pub fn gen_cluster(
        &self,
        generator: &OverworldGenerator,
        structure_seed: u64,
        center_pos: I64Vec3,
        cluster: &mut Cluster,
        structure_cache: Arc<OnceCell<Box<dyn StructureCache>>>,
    ) {
        (self.gen_fn)(
            self,
            generator,
            structure_seed,
            center_pos,
            cluster,
            structure_cache,
        );
    }
}

pub struct StructuresIter<'a> {
    pub(super) generator: &'a OverworldGenerator,
    pub(super) structure: &'a Structure,
    pub(super) start_cluster_pos: I64Vec3,
    pub(super) max_search_radius: u32,
    pub(super) queue: VecDeque<I64Vec3>,
    pub(super) traversed_nodes: BitVec,
}

impl Iterator for StructuresIter<'_> {
    type Item = I64Vec3;

    fn next(&mut self) -> Option<Self::Item> {
        let diam = self.max_search_radius as i64 * 2 - 1;
        let clusters_per_octant = self.structure.avg_spacing() as i64;

        let front = self.queue.front().unwrap();
        {
            let rel_pos = (front - self.start_cluster_pos).add_scalar(diam / 2);
            let idx_1d = (rel_pos.x * diam * diam + rel_pos.y * diam + rel_pos.x) as usize;
            self.traversed_nodes.set(idx_1d, true);

            let pos_in_clusters = front * clusters_per_octant;
            let (result, success) = self.generator.gen_structure_pos(self.structure, pos_in_clusters);

            if success {
                return Some(result);
            }
        }

        // Use breadth-first search
        while let Some(curr_pos) = self.queue.pop_front() {
            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let next_pos = curr_pos + dir;

                let rel_pos = (curr_pos - self.start_cluster_pos).add_scalar(diam / 2);
                if rel_pos.x < 0
                    || rel_pos.y < 0
                    || rel_pos.z < 0
                    || rel_pos.x >= diam
                    || rel_pos.y >= diam
                    || rel_pos.z >= diam
                {
                    continue;
                }

                let idx_1d = (rel_pos.x * diam * diam + rel_pos.y * diam + rel_pos.x) as usize;
                if self.traversed_nodes.get(idx_1d).unwrap() {
                    continue;
                }

                self.queue.push_back(next_pos);
                self.traversed_nodes.set(idx_1d, true);

                let pos_in_clusters = next_pos * clusters_per_octant;
                let (result, success) = self.generator.gen_structure_pos(self.structure, pos_in_clusters);

                if success {
                    return Some(result);
                }
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.max_search_radius.pow(3) as usize))
    }
}
