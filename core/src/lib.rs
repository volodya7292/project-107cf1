use nalgebra_glm as glm;

use crate::overworld::block::event_handlers::AfterTickActionsBuilder;
use crate::overworld::facing::Facing;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::BlockPos;
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl};
use crate::overworld::Overworld;

pub mod main_registry;
pub mod overworld;
// pub mod overworld_orchestrator;
// pub mod overworld_renderer;
pub mod physics;
pub mod registry;
pub mod scene;

pub fn on_liquid_tick(
    pos: &BlockPos,
    block_data: BlockData,
    overworld: &Overworld,
    mut result: AfterTickActionsBuilder,
) {
    let registry = overworld.main_registry().registry();
    let mut access = overworld.access();

    let curr_liquid = block_data.liquid_state();
    let mut can_be_deactivated = true;

    // Propagate liquid downwards
    let bottom_pos = pos.offset_i32(&Facing::NegativeY.direction());
    if let Some(bottom_data) = access.get_block(&bottom_pos) {
        let block = registry.get_block(bottom_data.block_id()).unwrap();

        if block.can_pass_liquid() {
            result.set_liquid(bottom_pos, LiquidState::max(curr_liquid.liquid_id()));
            result.set_activity(bottom_pos, true);
        }
    } else {
        can_be_deactivated = false;
    };

    // Propagate liquid sideways
    for dir in Facing::XZ_DIRECTIONS {
        let rel_pos = pos.offset_i32(&dir);

        let Some(rel_data) = access.get_block(&rel_pos) else {
            can_be_deactivated = false;
            continue;
        };

        // if rel_liquid

        // access.update_block(&rel_pos, |data| {});
    }

    if can_be_deactivated {
        result.set_activity(*pos, false);
    }
}
