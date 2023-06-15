use std::sync::Arc;

use crate::overworld::accessor::ReadOnlyOverworldAccessor;
use crate::overworld::actions_storage::OverworldActionsBuilder;
use crate::overworld::block::BlockData;
use crate::overworld::position::BlockPos;
use crate::overworld::Overworld;
use crate::registry::Registry;

/// Collects actions to perform after the tick.
/// Returns whether to keep the block at `pos` active.
pub type OnTickFn = fn(
    tick: u64,
    pos: &BlockPos,
    block_data: BlockData,
    registry: &Arc<Registry>,
    accessor: &mut ReadOnlyOverworldAccessor,
    result: OverworldActionsBuilder,
) -> bool;

/// Gets called when nearby block is set.
pub type OnNearbyBlockSet = fn(
    block_data: BlockData,
    near_by_block: BlockData,
    overworld: &Overworld,
    after_tick_actions: OverworldActionsBuilder,
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
