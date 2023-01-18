use crate::overworld::accessor::OverworldAccessor;
use crate::overworld::block::BlockState;
use crate::overworld::light_state::LightState;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::BlockPos;
use entity_data::{ArchetypeState, Component};
use std::{mem, ptr};

/// Returns true if the specified data was successfully applied.
pub type ApplyFn = fn(access: &mut OverworldAccessor, pos: &BlockPos, data: *const u8);

/// Drops underlying data.
pub type DropDataFn = fn(data: *mut u8);

pub struct StateChangeInfo {
    pub pos: BlockPos,
    pub data_ptr: *mut u8,
    pub apply_fn: ApplyFn,
    pub drop_data_fn: DropDataFn,
}

unsafe impl Send for StateChangeInfo {}

unsafe impl Sync for StateChangeInfo {}

impl StateChangeInfo {
    /// Safety: referenced data must be accessible and valid.
    pub unsafe fn apply(&self, access: &mut OverworldAccessor) {
        (self.apply_fn)(access, &self.pos, self.data_ptr)
    }
}

pub struct ActivityChangeInfo {
    pub pos: BlockPos,
    pub active: bool,
}

pub struct LightChangeInfo {
    pub pos: BlockPos,
    pub light: LightState,
}

pub struct LiquidChangeInfo {
    pub pos: BlockPos,
    pub liquid: LiquidState,
}

unsafe fn drop_typed<T>(data: *mut u8) {
    ptr::drop_in_place(data as *mut T);
}

/// Contains actions to perform after the tick.
#[derive(Default)]
pub struct OverworldActionsStorage {
    pub states: bumpalo::Bump,
    pub components: bumpalo::Bump,

    pub states_infos: Vec<StateChangeInfo>,
    pub components_infos: Vec<StateChangeInfo>,
    pub liquid_infos: Vec<LiquidChangeInfo>,
    pub set_light_infos: Vec<LightChangeInfo>,
    pub remove_light_positions: Vec<BlockPos>,
    pub activity_infos: Vec<ActivityChangeInfo>,
}

impl OverworldActionsStorage {
    pub fn new() -> Self {
        Self {
            states: Default::default(),
            components: Default::default(),
            states_infos: Vec::with_capacity(4096),
            components_infos: Vec::with_capacity(4096),
            liquid_infos: Vec::with_capacity(4096),
            set_light_infos: Vec::with_capacity(4096),
            remove_light_positions: Vec::with_capacity(4096),
            activity_infos: Vec::with_capacity(4096),
        }
    }

    pub fn clear(&mut self) {
        for info in self.states_infos.iter_mut().chain(&mut self.components_infos) {
            (info.drop_data_fn)(info.data_ptr);
        }

        self.states.reset();
        self.components.reset();
    }

    pub fn set_block<A: ArchetypeState>(&mut self, pos: BlockPos, block_state: BlockState<A>) {
        let mut_ref = self.states.alloc(block_state);

        self.states_infos.push(StateChangeInfo {
            pos,
            data_ptr: mut_ref as *mut _ as *mut u8,
            apply_fn: |access, pos, data| {
                let mut state_uninit = mem::MaybeUninit::<BlockState<A>>::uninit();

                let state = unsafe {
                    (data as *const BlockState<A>)
                        .copy_to_nonoverlapping(state_uninit.as_mut_ptr(), mem::size_of::<BlockState<A>>());
                    state_uninit.assume_init()
                };

                access.update_block(pos, |data| data.set(state));
            },
            drop_data_fn: |p| unsafe { drop_typed::<BlockState<A>>(p) },
        });
    }

    pub fn set_component<C: Component>(&mut self, pos: BlockPos, component: C) {
        let mut_ref = self.components.alloc(component);

        self.components_infos.push(StateChangeInfo {
            pos,
            data_ptr: mut_ref as *mut _ as *mut u8,
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
            drop_data_fn: |p| unsafe { drop_typed::<C>(p) },
        });
    }

    pub fn set_active(&mut self, pos: BlockPos, active: bool) {
        self.activity_infos.push(ActivityChangeInfo { pos, active });
    }

    pub fn set_light(&mut self, pos: BlockPos, light: LightState) {
        self.set_light_infos.push(LightChangeInfo { pos, light });
    }

    pub fn remove_light(&mut self, pos: BlockPos) {
        self.remove_light_positions.push(pos);
    }

    pub fn set_liquid(&mut self, pos: BlockPos, liquid: LiquidState) {
        self.liquid_infos.push(LiquidChangeInfo { pos, liquid });
    }

    pub fn builder(&mut self) -> OverworldActionsBuilder {
        OverworldActionsBuilder { storage: self }
    }
}

pub struct OverworldActionsBuilder<'a> {
    storage: &'a mut OverworldActionsStorage,
}

impl OverworldActionsBuilder<'_> {
    pub fn set_block<A: ArchetypeState>(&mut self, pos: BlockPos, block_state: BlockState<A>) {
        self.storage.set_block(pos, block_state);
    }

    pub fn set_component<C: Component>(&mut self, pos: BlockPos, component: C) {
        self.storage.set_component(pos, component);
    }

    pub fn set_light(&mut self, pos: BlockPos, light: LightState) {
        self.storage.set_light(pos, light);
    }

    pub fn remove_light(&mut self, pos: BlockPos) {
        self.storage.remove_light(pos);
    }

    pub fn set_liquid(&mut self, pos: BlockPos, liquid: LiquidState) {
        self.storage.set_liquid(pos, liquid);
    }

    pub fn set_activity(&mut self, pos: BlockPos, active: bool) {
        self.storage.set_active(pos, active);
    }
}
