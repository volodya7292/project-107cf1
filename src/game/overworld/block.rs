use std::ops::Range;

use approx::AbsDiffEq;
use nalgebra_glm as glm;
use nalgebra_glm::{BVec3, Vec2, Vec3};

use crate::game::overworld::block_component::Facing;
use crate::utils::{AllSame, AllSameBy};

#[derive(Default)]
pub struct BlockProps {
    textured_model_id: u32,
}

impl BlockProps {
    pub fn textured_model_id(&self) -> u32 {
        self.textured_model_id
    }
}
