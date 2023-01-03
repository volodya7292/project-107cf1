use nalgebra_glm as glm;

use crate::overworld::accessor::{
    OverworldAccessor, ReadOnlyOverworldAccessor, ReadOnlyOverworldAccessorImpl,
};
use crate::overworld::block::event_handlers::{AfterTickActionsBuilder, AfterTickActionsStorage};
use crate::overworld::facing::Facing;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::position::BlockPos;
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl};
use crate::overworld::{Overworld, ReadOnlyOverworld};

pub mod main_registry;
pub mod overworld;
// pub mod overworld_orchestrator;
// pub mod overworld_renderer;
pub mod physics;
pub mod registry;
pub mod scene;

pub fn on_liquid_tick(
    tick: u64,
    pos: &BlockPos,
    block_data: BlockData,
    overworld: ReadOnlyOverworld<'_>,
    access: &mut ReadOnlyOverworldAccessor,
    mut result: AfterTickActionsBuilder,
) {
    if tick % 10 != 0 {
        return;
    }

    let registry = overworld.main_registry().registry();
    let curr_liquid = block_data.liquid_state();

    // Propagate liquid downwards
    let bottom_pos = pos.offset_i32(&Facing::NegativeY.direction());
    if let Some(bottom_data) = access.get_block(&bottom_pos) {
        let block = registry.get_block(bottom_data.block_id()).unwrap();

        if block.can_pass_liquid() {
            result.set_liquid(bottom_pos, LiquidState::max(curr_liquid.liquid_id()));
        }
    } else {
        return;
    };

    let mut can_spread = curr_liquid.level() > 2; // TODO: adjust minimum spread level
    let mut do_spread = curr_liquid.is_source();

    if can_spread && !do_spread {
        // Check horizontal directions
        for dir in &Facing::XZ_DIRECTIONS {
            let rel_pos = pos.offset_i32(dir);
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return;
            };
            if rel_data.liquid_state().level() >= curr_liquid.level() {
                // if rel_data.liquid_state().level() == (curr_liquid.level() + 1)
                do_spread = true;
                break;
            }
        }

        // Check top
        if !do_spread {
            let rel_pos = pos.offset_i32(Facing::PositiveY.direction());
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return;
            };
            if rel_data.liquid_state().level() > 0 {
                do_spread = true;
            }
        }
    }

    if can_spread && do_spread {
        let spread_liquid = LiquidState::new(curr_liquid.liquid_id(), curr_liquid.level() - 2); // TODO: change 2 to the speed
        let bottom_is_solid;

        // Process bottom and check it for solidness
        {
            let rel_pos = pos.offset_i32(&Facing::NegativeY.direction());
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return;
            };
            let block = registry.get_block(rel_data.block_id()).unwrap();
            if block.can_pass_liquid() && rel_data.liquid_state().level() < spread_liquid.level() {
                result.set_liquid(rel_pos, LiquidState::max(curr_liquid.liquid_id()));
            }
            bottom_is_solid = !block.can_pass_liquid();
        }

        if bottom_is_solid {
            for dir in Facing::XZ_DIRECTIONS {
                let rel_pos = pos.offset_i32(&dir);
                let Some(rel_data) = access.get_block(&rel_pos) else {
                    return;
                };
                let block = registry.get_block(rel_data.block_id()).unwrap();

                if block.can_pass_liquid() && rel_data.liquid_state().level() < spread_liquid.level() {
                    result.set_liquid(rel_pos, spread_liquid);
                }
            }
        }
    }

    if !can_spread || do_spread {
        result.set_activity(*pos, false);
    }
}

/*

current block props:
    curr_liquid_level

if the block is a source:
    spread further
    deactivate itself
else:
    if a least one neighbour has neighbour_liquid_level == curr_liquid_level + 1:
        spread (curr_liquid_level - 1)
        deactivate itself
    else:
        curr_liquid_level -= 1;
        activate neighbours
        if curr_liquid_level == 0:
            deactivate itself

 */
