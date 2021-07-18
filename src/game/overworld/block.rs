use std::ops::Range;

use approx::AbsDiffEq;
use nalgebra_glm as glm;
use nalgebra_glm::{BVec3, Vec2, Vec3};

use crate::game::overworld::block_component::Facing;
use crate::utils::{AllSame, AllSameBy};

#[derive(Copy, Clone)]
pub struct Block {
    archetype: u32,
    textured_model: u32,
    is_none: bool,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            archetype: u32::MAX,
            textured_model: u32::MAX,
            is_none: true,
        }
    }
}

impl Block {
    pub fn new(archetype: u32, textured_model: u32) -> Block {
        Block {
            archetype,
            textured_model,
            is_none: false,
        }
    }

    pub fn archetype(&self) -> u32 {
        self.archetype
    }

    pub fn textured_model(&self) -> u32 {
        self.textured_model
    }

    pub fn is_none(&self) -> bool {
        self.is_none
    }
}
