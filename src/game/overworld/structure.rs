use crate::game::overworld::{cluster, Overworld};
use engine::utils::UInt;
use nalgebra_glm::{I64Vec3, U64Vec3};

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
