use std::ops::Div;

use nalgebra_glm as glm;
use nalgebra_glm::{I64Vec3, U64Vec3};
use rand::Rng;

use crate::game::overworld::Overworld;
use crate::game::overworld_streamer;
use crate::utils;
use crate::utils::{Integer, UInteger};

pub mod world;

/// Returns whether the structure can be generated at specified center position.
pub type GenPosFn = fn(structure: &Structure, overworld: &Overworld, center_pos: I64Vec3) -> bool;

pub struct Structure {
    max_size: U64Vec3,
    /// cluster level that fits structure size
    cluster_level: u32,
    /// spacing in clusters of level `cluster_level`
    min_spacing: u64,
    avg_spacing: u64,
    gen_fn: GenPosFn,
}

impl Structure {
    pub fn new(max_size: U64Vec3, min_spacing: u64, avg_spacing: u64, gen_fn: GenPosFn) -> Structure {
        assert!(avg_spacing > min_spacing);

        let cluster_level = max_size.max().next_power_of_two().log2() as u32;

        Structure {
            max_size,
            cluster_level,
            min_spacing,
            avg_spacing,
            gen_fn,
        }
    }

    pub fn cluster_level(&self) -> u32 {
        self.cluster_level
    }

    pub fn min_spacing(&self) -> u64 {
        self.min_spacing
    }

    pub fn avg_spacing(&self) -> u64 {
        self.avg_spacing
    }

    pub fn check_gen_pos(&self, overworld: &Overworld, center_pos: I64Vec3) -> bool {
        (self.gen_fn)(self, overworld, center_pos)
    }
}
