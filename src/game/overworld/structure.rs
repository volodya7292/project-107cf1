use crate::game::overworld::block_component::Facing;
use crate::game::overworld::{cluster, Overworld};
use crate::glm;
use bit_vec::BitVec;
use engine::utils::UInt;
use nalgebra_glm::{I64Vec3, U64Vec3};
use std::collections::VecDeque;

pub mod world;

/// Returns whether the structure can be generated at specified center position.
pub type GenPosCheckFn = fn(structure: &Structure, overworld: &Overworld, center_pos: I64Vec3) -> bool;

pub struct Structure {
    uid: u32,
    /// Maximum size in blocks
    max_size: U64Vec3,
    /// Minimum spacing in clusters between structures of this type
    min_spacing: u64,
    /// Average spacing in clusters between structures of this type
    avg_spacing: u64,
    gen_pos_check_fn: GenPosCheckFn,
}

impl Structure {
    /// `uid` must be unique to avoid generating different structures at the same cluster
    pub fn new(
        uid: u32,
        max_size: U64Vec3,
        min_spacing: u64,
        avg_spacing: u64,
        gen_pos_check_fn: GenPosCheckFn,
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

    pub fn check_gen_pos(&self, overworld: &Overworld, center_pos: I64Vec3) -> bool {
        (self.gen_pos_check_fn)(self, overworld, center_pos)
    }
}

pub struct StructuresIter<'a> {
    pub(super) overworld: &'a Overworld,
    pub(super) structure: &'a Structure,
    pub(super) start_cluster_pos: I64Vec3,
    pub(super) max_search_radius: u32,
    pub(super) queue: VecDeque<I64Vec3>,
    pub(super) traversed_nodes: BitVec,
}

impl Iterator for StructuresIter<'_> {
    type Item = I64Vec3;

    fn next(&mut self) -> Option<Self::Item> {
        let diam = (self.max_search_radius * 2) as i64;
        let clusters_per_octant = self.structure.avg_spacing() as i64;

        while let Some(curr_pos) = self.queue.pop_front() {
            for i in 0..6 {
                let dir: I64Vec3 = glm::convert(Facing::DIRECTIONS[i]);
                let next_pos = curr_pos + dir;

                let rel_pos = (curr_pos - self.start_cluster_pos).add_scalar(self.max_search_radius as i64);
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
                let (result, success) = self.overworld.gen_structure_pos(self.structure, pos_in_clusters);

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
