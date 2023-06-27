pub mod event_handlers;

pub use crate::overworld::block::event_handlers::EventHandlers;
use crate::overworld::occluder::Occluder;
use crate::overworld::raw_cluster::BlockData;
use crate::registry::Registry;
use entity_data::{AnyState, ArchetypeState, EntityId, EntityStorage, StaticArchetype};
use serde::de::DeserializeOwned;
use serde::Serialize;

#[derive(Copy, Clone)]
pub struct Block {
    transparent: bool,
    model_id: u16,
    occluder: Occluder,
    event_handlers: EventHandlers,
    active_by_default: bool,
    can_pass_light: bool,
    can_pass_liquid: bool,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            transparent: false,
            model_id: u16::MAX,
            occluder: Default::default(),
            event_handlers: Default::default(),
            active_by_default: false,
            can_pass_light: false,
            can_pass_liquid: false,
        }
    }
}

impl Block {
    pub fn transparent(&self) -> bool {
        self.transparent
    }

    pub fn model_id(&self) -> u16 {
        self.model_id
    }

    pub fn occluder(&self) -> Occluder {
        self.occluder
    }

    pub fn event_handlers(&self) -> &EventHandlers {
        &self.event_handlers
    }

    pub fn is_opaque(&self) -> bool {
        self.occluder.is_full()
    }

    pub fn is_model_invisible(&self) -> bool {
        self.model_id == Registry::MODEL_ID_NULL
    }

    pub fn active_by_default(&self) -> bool {
        self.active_by_default
    }

    pub fn can_pass_light(&self) -> bool {
        self.can_pass_light
    }

    pub fn can_pass_liquid(&self) -> bool {
        self.can_pass_liquid
    }
}

#[derive(Copy, Clone)]
pub struct BlockBuilder {
    transparent: bool,
    active_by_default: bool,
    can_pass_light: bool,
    can_pass_liquid: bool,
    model_id: u16,
    event_handlers: EventHandlers,
}

impl BlockBuilder {
    pub fn new(model_id: u16) -> Self {
        BlockBuilder {
            transparent: false,
            active_by_default: false,
            can_pass_light: false,
            can_pass_liquid: false,
            model_id,
            event_handlers: Default::default(),
        }
    }

    pub fn with_transparent(mut self, transparent: bool) -> Self {
        self.transparent = transparent;
        self
    }

    pub fn with_event_handlers(mut self, event_handlers: EventHandlers) -> Self {
        self.event_handlers = event_handlers;
        self
    }

    pub fn with_active_by_default(mut self, active: bool) -> Self {
        self.active_by_default = active;
        self
    }

    pub fn with_can_pass_light(mut self, can_pass_light: bool) -> Self {
        self.can_pass_light = can_pass_light;
        self
    }

    pub fn with_can_pass_liquid(mut self, can_pass_liquid: bool) -> Self {
        self.can_pass_liquid = can_pass_liquid;
        self
    }

    pub fn build(self, registry: &Registry) -> Block {
        let mut occluder = Occluder::EMPTY;

        if !self.transparent {
            if let Some(model) = registry.get_block_model(self.model_id) {
                occluder = model.occluder();
            }
        }

        Block {
            transparent: self.transparent,
            model_id: self.model_id,
            occluder,
            event_handlers: self.event_handlers,
            active_by_default: self.active_by_default,
            can_pass_light: self.can_pass_light,
            can_pass_liquid: self.can_pass_liquid,
        }
    }

    pub fn model_id(&self) -> u16 {
        self.model_id
    }

    pub fn event_handlers(&self) -> &EventHandlers {
        &self.event_handlers
    }
}

#[derive(Copy)]
pub struct BlockState<A: ArchetypeState> {
    pub block_id: u16,
    pub components: A,
    clone_any_fn: fn(&A) -> A,
}

pub trait BlockStateArchetype: StaticArchetype + Serialize + DeserializeOwned {
    fn canon_name() -> &'static str;

    fn serialize_from(
        storage: &EntityStorage,
        entity_id: &EntityId,
        serializer: &mut dyn FnMut(&dyn erased_serde::Serialize),
    ) {
        let state = storage.get_state::<Self>(entity_id).unwrap();
        serializer(state);
    }

    fn deserialize_into(
        deserializer: &mut dyn erased_serde::Deserializer,
        storage: &mut EntityStorage,
    ) -> EntityId {
        let this = Self::deserialize(deserializer).unwrap();
        storage.add(this)
    }
}

impl<A: BlockStateArchetype + Clone> BlockState<A> {
    pub fn new(block_id: u16, state: A) -> Self {
        Self {
            block_id,
            components: state,
            clone_any_fn: |f: &A| f.clone(),
        }
    }
}

impl<A: StaticArchetype + Clone> BlockState<A> {
    pub fn into_any(self) -> BlockState<AnyState> {
        BlockState {
            block_id: self.block_id,
            components: self.components.into_any(),
            clone_any_fn: |f| f.downcast_ref::<A>().unwrap().clone().into_any(),
        }
    }
}

impl<A: ArchetypeState> Clone for BlockState<A> {
    fn clone(&self) -> Self {
        Self {
            block_id: self.block_id,
            components: (self.clone_any_fn)(&self.components),
            clone_any_fn: self.clone_any_fn,
        }
    }
}

pub type AnyBlockState = BlockState<AnyState>;

impl AnyBlockState {
    pub fn downcast<A, B>(self) -> Option<BlockState<B>>
    where
        A: ArchetypeState,
        B: StaticArchetype + Clone,
    {
        Some(BlockState {
            block_id: self.block_id,
            components: self.components.downcast()?,
            clone_any_fn: |f| f.clone(),
        })
    }
}
