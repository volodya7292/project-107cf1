use std::any::TypeId;
use std::mem;

use entity_data::{ArchetypeState, Component};
use nalgebra_glm::I64Vec3;
use smallvec::SmallVec;

use engine::unwrap_option;
use engine::utils::HashMap;

use crate::game::overworld::block::{BlockData, BlockState};
use crate::game::overworld::clusters_access_cache::ClustersAccessor;
use crate::game::overworld::raw_cluster::BlockDataImpl;
use crate::game::AnyBlockState;
use crate::game::Overworld;

/// Returns true if the specified state/component was successfully set.
pub type ApplyFn = fn(access: &mut ClustersAccessor, pos: &I64Vec3, data: *const u8) -> bool;

pub struct BlockStateInfo {
    pub ptr: *const u8,
    pub pos: I64Vec3,
    pub apply_fn: ApplyFn,
}
unsafe impl Send for BlockStateInfo {}
unsafe impl Sync for BlockStateInfo {}

pub struct BlockComponentInfo {
    pub ptr: *const u8,
    pub pos: I64Vec3,
    pub apply_fn: ApplyFn,
}
unsafe impl Send for BlockComponentInfo {}
unsafe impl Sync for BlockComponentInfo {}

/// Contains actions to perform after the tick.
pub struct AfterTickActionsStorage {
    pub states: bumpalo::Bump,
    pub components: bumpalo::Bump,

    pub states_infos: Vec<BlockStateInfo>,
    pub components_infos: Vec<BlockComponentInfo>,
}

impl AfterTickActionsStorage {
    pub fn new() -> Self {
        Self {
            states: Default::default(),
            components: Default::default(),
            states_infos: Vec::with_capacity(4096),
            components_infos: Vec::with_capacity(4096),
        }
    }

    pub fn set_block<A: ArchetypeState>(&mut self, pos: I64Vec3, block_state: BlockState<A>) {
        let mut_ref = self.states.alloc(block_state);

        self.states_infos.push(BlockStateInfo {
            ptr: mut_ref as *const _ as *const u8,
            pos,
            apply_fn: |access, pos, data| unsafe {
                let mut state = mem::MaybeUninit::<BlockState<A>>::uninit();

                (data as *const BlockState<A>)
                    .copy_to_nonoverlapping(state.as_mut_ptr(), mem::size_of::<BlockState<A>>());
                let state_init = state.assume_init();

                access.set_block(pos, state_init)
            },
        });
    }

    pub fn set_component<C: Component>(&mut self, pos: I64Vec3, component: C) {
        let mut_ref = self.components.alloc(component);

        self.components_infos.push(BlockComponentInfo {
            ptr: mut_ref as *const _ as *const u8,
            pos,
            apply_fn: |access, pos, data| unsafe {
                let mut component = mem::MaybeUninit::<C>::uninit();

                (data as *const C).copy_to_nonoverlapping(component.as_mut_ptr(), mem::size_of::<C>());
                let component_init = component.assume_init();

                access.update_block(pos, |data| {
                    if let Some(comp) = data.get_mut::<C>() {
                        *comp = component_init
                    }
                })
            },
        });
    }

    pub fn builder(&mut self) -> AfterTickActionsBuilder {
        AfterTickActionsBuilder { storage: self }
    }
}

pub struct AfterTickActionsBuilder<'a> {
    storage: &'a mut AfterTickActionsStorage,
}

impl AfterTickActionsBuilder<'_> {
    pub fn set_block<A: ArchetypeState>(&mut self, pos: I64Vec3, block_state: BlockState<A>) {
        self.storage.set_block(pos, block_state);
    }

    pub fn set_component<C: Component>(&mut self, pos: I64Vec3, component: C) {
        self.storage.set_component(pos, component);
    }
}

/// Returns the actions to perform after the tick.
pub type OnTickFn =
    fn(pos: &I64Vec3, block_data: BlockData, overworld: &Overworld, result: AfterTickActionsBuilder);

/// Gets called when nearby block is set.
pub type OnNearbyBlockSet = fn(
    block_data: BlockData,
    near_by_block: BlockData,
    overworld: &Overworld,
    after_tick_actions: AfterTickActionsBuilder,
);

#[derive(Copy, Clone, Default)]
pub struct EventHandlers {
    pub on_tick: Option<OnTickFn>,
    pub on_nearby_block_set: Option<OnTickFn>,
}

impl EventHandlers {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_on_tick(mut self, on_tick: OnTickFn) -> Self {
        self.on_tick = Some(on_tick);
        self
    }
}
