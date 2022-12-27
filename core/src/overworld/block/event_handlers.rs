use std::any::TypeId;
use std::mem;

use entity_data::{ArchetypeState, Component};
use nalgebra_glm::I64Vec3;
use smallvec::SmallVec;
use winit::event::VirtualKeyCode::P;

use engine::unwrap_option;
use engine::utils::HashMap;

use crate::overworld::accessor::{OverworldAccessor, ReadOnlyOverworldAccessor};
use crate::overworld::block::{BlockData, BlockState};
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::BlockPos;
use crate::overworld::raw_cluster::BlockDataImpl;
use crate::overworld::Overworld;

/// Returns true if the specified data was successfully applied.
pub type ApplyFn = fn(access: &mut OverworldAccessor, pos: &BlockPos, data: *const u8);

pub struct StateChangeInfo {
    pub pos: BlockPos,
    pub data_ptr: *const u8,
    pub apply_fn: ApplyFn,
}
unsafe impl Send for StateChangeInfo {}
unsafe impl Sync for StateChangeInfo {}

impl StateChangeInfo {
    /// Safety: referenced data must be accessible and valid.
    pub unsafe fn apply(&self, access: &mut OverworldAccessor) {
        (self.apply_fn)(access, &self.pos, self.data_ptr)
    }
}

/// Contains actions to perform after the tick.
pub struct AfterTickActionsStorage {
    pub states: bumpalo::Bump,
    pub components: bumpalo::Bump,
    pub activities: bumpalo::Bump,
    pub liquids: bumpalo::Bump,

    pub states_infos: Vec<StateChangeInfo>,
    pub components_infos: Vec<StateChangeInfo>,
    pub activity_infos: Vec<StateChangeInfo>,
    pub liquid_infos: Vec<StateChangeInfo>,
}

impl AfterTickActionsStorage {
    pub fn new() -> Self {
        Self {
            states: Default::default(),
            components: Default::default(),
            activities: Default::default(),
            liquids: Default::default(),
            states_infos: Vec::with_capacity(4096),
            components_infos: Vec::with_capacity(4096),
            activity_infos: Vec::with_capacity(4096),
            liquid_infos: Vec::with_capacity(4096),
        }
    }

    pub fn set_block<A: ArchetypeState>(&mut self, pos: BlockPos, block_state: BlockState<A>) {
        let mut_ref = self.states.alloc(block_state);

        self.states_infos.push(StateChangeInfo {
            pos,
            data_ptr: mut_ref as *const _ as *const u8,
            apply_fn: |access, pos, data| {
                let mut state_uninit = mem::MaybeUninit::<BlockState<A>>::uninit();

                let state = unsafe {
                    (data as *const BlockState<A>)
                        .copy_to_nonoverlapping(state_uninit.as_mut_ptr(), mem::size_of::<BlockState<A>>());
                    state_uninit.assume_init()
                };

                access.update_block(pos, |data| data.set(state));
            },
        });
    }

    pub fn set_component<C: Component>(&mut self, pos: BlockPos, component: C) {
        let mut_ref = self.components.alloc(component);

        self.components_infos.push(StateChangeInfo {
            pos,
            data_ptr: mut_ref as *const _ as *const u8,
            apply_fn: |access, pos, data| {
                let mut component_uninit = mem::MaybeUninit::<C>::uninit();

                let component = unsafe {
                    (data as *const C)
                        .copy_to_nonoverlapping(component_uninit.as_mut_ptr(), mem::size_of::<C>());
                    component_uninit.assume_init()
                };

                access.update_block(pos, |data| {
                    if let Some(comp) = data.get_mut::<C>() {
                        *comp = component
                    }
                });
            },
        });
    }

    pub fn set_active(&mut self, pos: BlockPos, active: bool) {
        let mut_ref = self.activities.alloc(active);

        self.activity_infos.push(StateChangeInfo {
            data_ptr: mut_ref as *const _ as *const u8,
            pos,
            apply_fn: |access: &mut OverworldAccessor, pos: &BlockPos, data: *const u8| {
                let active = unsafe { *(data as *const bool) };
                access.update_block(pos, |data| {
                    *data.active_mut() = active;
                });
            },
        });
    }

    pub fn set_liquid(&mut self, pos: BlockPos, liquid: LiquidState) {
        let mut_ref = self.liquids.alloc(liquid);

        self.liquid_infos.push(StateChangeInfo {
            pos,
            data_ptr: mut_ref as *const _ as *const u8,
            apply_fn: |access: &mut OverworldAccessor, pos: &BlockPos, data: *const u8| {
                let liquid = unsafe { *(data as *const LiquidState) };
                access.set_liquid_state(pos, liquid);
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
    pub fn set_block<A: ArchetypeState>(&mut self, pos: BlockPos, block_state: BlockState<A>) {
        self.storage.set_block(pos, block_state);
    }

    pub fn set_component<C: Component>(&mut self, pos: BlockPos, component: C) {
        self.storage.set_component(pos, component);
    }

    pub fn set_liquid(&mut self, pos: BlockPos, liquid: LiquidState) {
        self.storage.set_liquid(pos, liquid);
    }

    pub fn set_activity(&mut self, pos: BlockPos, active: bool) {
        self.storage.set_active(pos, active);
    }
}

/// Returns the actions to perform after the tick.
pub type OnTickFn = fn(
    pos: &BlockPos,
    block_data: BlockData,
    overworld: &Overworld,
    accessor: &mut ReadOnlyOverworldAccessor,
    result: AfterTickActionsBuilder,
);

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
