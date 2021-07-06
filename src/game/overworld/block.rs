use std::ops::Range;

use approx::AbsDiffEq;
use nalgebra_glm as glm;
use nalgebra_glm::{BVec3, Vec2, Vec3};

use crate::game::overworld::block_component::Facing;
use crate::utils::{AllSame, AllSameBy};

pub struct BlockProps {
    pub textured_model_id: u32,
    pub is_none: bool,
}

impl Default for BlockProps {
    fn default() -> Self {
        BlockProps {
            textured_model_id: u32::MAX,
            is_none: true,
        }
    }
}

impl BlockProps {
    pub fn textured_model_id(&self) -> u32 {
        self.textured_model_id
    }
}
