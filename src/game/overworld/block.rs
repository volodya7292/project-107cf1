use std::ops::Range;

use approx::AbsDiffEq;
use nalgebra_glm as glm;
use nalgebra_glm::{BVec3, Vec2, Vec3};

use crate::game::overworld::block_component::Facing;
use crate::utils::{AllSame, AllSameBy};

#[derive(Copy, Clone)]
pub struct Block {
    archetype: u16,
    textured_model: u16,
    is_empty: bool,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            archetype: u16::MAX,
            textured_model: u16::MAX,
            is_empty: true,
        }
    }
}

impl Block {
    pub fn new(archetype: u16, textured_model: u16) -> Block {
        Block {
            archetype,
            textured_model,
            is_empty: false,
        }
    }

    pub fn archetype(&self) -> u16 {
        self.archetype
    }

    pub fn textured_model(&self) -> u16 {
        self.textured_model
    }

    pub fn is_empty(&self) -> bool {
        self.is_empty
    }
}
