use entity_data::{AnyState, ArchetypeState, StaticArchetype};
use std::any::Any;

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Block {
    textured_model: u16,
    on_tick: Option<fn()>,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            textured_model: u16::MAX,
            on_tick: None,
        }
    }
}

impl Block {
    pub fn new(textured_model: u16) -> Block {
        Block {
            textured_model,
            on_tick: None,
        }
    }

    pub fn textured_model(&self) -> u16 {
        self.textured_model
    }

    pub fn has_textured_model(&self) -> bool {
        self.textured_model != u16::MAX
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub struct BlockState<T: ArchetypeState> {
    pub block_id: u16,
    pub components: T,
}

impl<A: ArchetypeState + 'static> BlockState<A> {
    pub fn new(block_id: u16, state: A) -> Self {
        Self {
            block_id,
            components: state,
        }
    }

    pub fn into_any(self) -> BlockState<AnyState> {
        BlockState {
            block_id: self.block_id,
            components: self.components.into_any(),
        }
    }
}

pub type AnyBlockState = BlockState<AnyState>;

impl AnyBlockState {
    pub fn into_definite<A>(self) -> Option<BlockState<A>>
    where
        A: StaticArchetype + ArchetypeState + 'static,
    {
        Some(BlockState {
            block_id: self.block_id,
            components: self.components.into_definite()?,
        })
    }
}
