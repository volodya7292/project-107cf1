pub mod world;

use crate::game::overworld::{streamer, Overworld};
use crate::utils;
use crate::utils::{Integer, UInteger};
use nalgebra_glm as glm;
use nalgebra_glm::{I64Vec3, U64Vec3};
use rand::Rng;
use std::ops::Div;

/// Returns whether the structure can be generated at specified center position.
pub type GenPosFn = fn(structure: &Structure, overworld: &Overworld, center_pos: I64Vec3) -> bool;

pub struct Structure {
    size: U64Vec3,
    /// cluster level that fits structure size
    cluster_level: u32,
    /// spacing in clusters of level `cluster_level`
    min_spacing: u64,
    avg_spacing: u64,
    gen_fn: GenPosFn,
}

impl Structure {
    pub fn new(min_spacing: u64, avg_spacing: u64, gen_fn: GenPosFn) -> Structure {
        assert!(avg_spacing > min_spacing);

        let size: U64Vec3 = Default::default();
        let max_lod = size.max().next_power_of_two().log2() as u32;

        Structure {
            size: Default::default(),
            cluster_level: max_lod,
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

    /// Returns position of the structure center within gen-octant corresponding to the cluster position
    /// if the structure can be generated in the gen-octant.
    ///
    /// `cluster_pos` is a cluster position of level `Structure::cluster_level`.
    pub fn gen_pos(&self, overworld: &Overworld, cluster_pos: I64Vec3) -> Option<I64Vec3> {
        let structure_fit_size = streamer::cluster_size(self.cluster_level);
        let octant_pos = cluster_pos.map(|v| v.div_floor(self.avg_spacing as i64));
        let octant_pos_u64 = octant_pos.map(|v| u64::from_ne_bytes(v.to_ne_bytes()));
        let octant_size = self.avg_spacing * structure_fit_size;

        let mut rng = overworld
            .value_noise
            .state()
            .next(self.cluster_level as u64)
            .next(octant_pos_u64.x)
            .next(octant_pos_u64.y)
            .next(octant_pos_u64.z)
            .rng();
        let present = rng.gen::<bool>();

        if present {
            let r = (structure_fit_size / 2)..(octant_size - structure_fit_size / 2);
            let dx = rng.gen_range(r.clone());
            let dy = rng.gen_range(r.clone());
            let dz = rng.gen_range(r.clone());
            let center_pos =
                octant_pos * (octant_size as i64) + I64Vec3::new(dx as i64, dy as i64, dz as i64);

            (self.gen_fn)(self, overworld, center_pos).then(|| center_pos)
        } else {
            None
        }
    }
}
