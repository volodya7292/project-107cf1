use std::any::Any;

use entity_data::{AnyState, ArchetypeState, StaticArchetype};
use nalgebra_glm::I64Vec3;
use smallvec::SmallVec;

pub use crate::game::overworld::block::event_handlers::EventHandlers;
use crate::game::overworld::occluder::Occluder;
use crate::game::overworld::raw_cluster::BlockData;
use crate::game::overworld::Overworld;
use crate::game::registry::Registry;

pub mod event_handlers;
pub mod water;

#[derive(Copy, Clone)]
pub struct Block {
    textured_model: u16,
    occluder: Occluder,
    event_handlers: EventHandlers,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            textured_model: u16::MAX,
            occluder: Default::default(),
            event_handlers: Default::default(),
        }
    }
}

impl Block {
    pub fn new(registry: &Registry, textured_model: u16, event_handlers: EventHandlers) -> Block {
        let model = registry.get_textured_block_model(textured_model);
        Block {
            textured_model,
            occluder: model.map_or(Default::default(), |m| m.occluder()),
            event_handlers,
        }
    }

    pub fn new_simple(registry: &Registry, textured_model: u16) -> Block {
        Self::new(registry, textured_model, Default::default())
    }

    pub fn textured_model(&self) -> u16 {
        self.textured_model
    }

    pub fn has_textured_model(&self) -> bool {
        self.textured_model != u16::MAX
    }

    pub fn occluder(&self) -> Occluder {
        self.occluder
    }

    pub fn event_handlers(&self) -> &EventHandlers {
        &self.event_handlers
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BlockState<A: ArchetypeState> {
    pub block_id: u16,
    pub components: A,
}

impl<A: ArchetypeState> BlockState<A> {
    pub fn new(block_id: u16, state: A) -> Self {
        Self {
            block_id,
            components: state,
        }
    }
}

impl<A: StaticArchetype> BlockState<A> {
    pub fn into_any(self) -> BlockState<AnyState> {
        BlockState {
            block_id: self.block_id,
            components: self.components.into_any(),
        }
    }
}

pub type AnyBlockState = BlockState<AnyState>;

impl AnyBlockState {
    pub fn into_definite<A, B>(self) -> Option<BlockState<B>>
    where
        A: ArchetypeState,
        B: StaticArchetype,
    {
        Some(BlockState {
            block_id: self.block_id,
            components: self.components.into_static()?,
        })
    }
}
