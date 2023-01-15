use std::collections::hash_map;
use std::sync::Arc;
use std::time::Duration;

use nalgebra_glm as glm;
pub use once_cell;
use parking_lot::Mutex;
use rayon::prelude::*;

pub use macos;
use overworld::actions_storage::{OverworldActionsBuilder, OverworldActionsStorage, StateChangeInfo};

use crate::overworld::accessor::{
    OverworldAccessor, ReadOnlyOverworldAccessor, ReadOnlyOverworldAccessorImpl,
};
use crate::overworld::facing::Facing;
use crate::overworld::light_state::LightState;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::orchestrator::{OverworldOrchestrator, OverworldUpdateResult};
use crate::overworld::position::{BlockPos, ClusterPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl};
use crate::overworld::LoadedClusters;
use crate::registry::Registry;
use crate::utils::{HashMap, MO_RELAXED};

pub mod main_registry;
pub mod overworld;
// pub mod overworld_orchestrator;
// pub mod overworld_renderer;
pub mod execution;
pub mod physics;
pub mod registry;
pub mod scene;
pub mod utils;

/*

Liquid spread algorithm:

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
pub fn on_liquid_tick(
    tick: u64,
    pos: &BlockPos,
    block_data: BlockData,
    registry: &Registry,
    access: &mut ReadOnlyOverworldAccessor,
    mut result: OverworldActionsBuilder,
) {
    if tick % 10 != 0 {
        return;
    }

    let curr_liquid = block_data.liquid_state();

    if curr_liquid.is_empty() {
        result.set_activity(*pos, false);
        return;
    }

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

    const SPREAD_SPEED: u8 = 2;
    let can_spread = curr_liquid.level() > SPREAD_SPEED; // TODO: adjust minimum spread level
    let mut has_source = curr_liquid.is_source();
    let mut max_xz_neighbour_level = 0;

    // Make further checks to determine `has_source`
    if !has_source {
        // Check horizontal directions
        for dir in &Facing::XZ_DIRECTIONS {
            let rel_pos = pos.offset_i32(dir);
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return;
            };
            let rel_liquid = rel_data.liquid_state();

            if rel_liquid.level() > curr_liquid.level() {
                has_source = true;
            }

            max_xz_neighbour_level = max_xz_neighbour_level.max(rel_liquid.level());
        }

        // Check top
        if !has_source {
            let rel_pos = pos.offset_i32(Facing::PositiveY.direction());
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return;
            };
            if rel_data.liquid_state().level() > 0 {
                has_source = true;
            }
        }
    }

    // If the liquid doesn't have neighbouring source, is must vanish
    if !has_source {
        let new_liquid = LiquidState::new(
            curr_liquid.liquid_id(),
            max_xz_neighbour_level.min(curr_liquid.level().saturating_sub(SPREAD_SPEED)),
        );
        result.set_liquid(*pos, new_liquid);

        for dir in Facing::XZ_DIRECTIONS
            .iter()
            .chain([Facing::NegativeY.direction()])
        {
            let rel_pos = pos.offset_i32(dir);
            result.set_activity(rel_pos, true);
        }

        return;
    }

    // Spread the liquid horizontally
    if has_source && can_spread {
        let spread_liquid = LiquidState::new(curr_liquid.liquid_id(), curr_liquid.level() - SPREAD_SPEED);
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

    if !can_spread || has_source {
        result.set_activity(*pos, false);
    }
}

pub fn process_active_blocks(
    tick: u64,
    registry: &Arc<Registry>,
    loaded_clusters: &LoadedClusters,
) -> Vec<OverworldActionsStorage> {
    let total_after_actions = Mutex::new(Vec::with_capacity(loaded_clusters.read().len()));

    loaded_clusters.read().par_iter().for_each(|(cl_pos, o_cluster)| {
        if !o_cluster.has_active_blocks.load(MO_RELAXED) {
            return;
        }

        let cluster = o_cluster.cluster.read();
        let mut after_actions = OverworldActionsStorage::new();

        if let Some(cluster) = &*cluster {
            let mut accessor =
                OverworldAccessor::new(Arc::clone(registry), Arc::clone(loaded_clusters)).into_read_only();

            for (pos, block_data) in cluster.active_blocks() {
                let global_pos = cl_pos.to_block_pos().offset(&glm::convert(*pos.get()));
                let block = registry.get_block(block_data.block_id()).unwrap();

                // Check for liquid
                on_liquid_tick(
                    tick,
                    &global_pos,
                    block_data,
                    &registry,
                    &mut accessor,
                    after_actions.builder(),
                );

                // Block-specific on-tick event
                if let Some(on_tick) = &block.event_handlers().on_tick {
                    on_tick(
                        tick,
                        &global_pos,
                        block_data,
                        &registry,
                        &mut accessor,
                        after_actions.builder(),
                    );
                }
            }
        }

        total_after_actions.lock().push(after_actions);
    });

    total_after_actions.into_inner()
}

/// Used to propagate lighting and liquids from neighbours when the block at `block_pos` is removed.
fn check_block(registry: &Registry, access: &mut OverworldAccessor, pos: &BlockPos) {
    let curr_data = access.get_block(pos).unwrap();
    let curr_block = registry.get_block(curr_data.block_id()).unwrap();

    if !curr_block.can_pass_light() {
        access.remove_light(pos);
    }
    if !curr_block.can_pass_liquid() {
        access.update_block(pos, |data| {
            *data.liquid_state_mut() = LiquidState::NONE;
        });
    }

    for dir in &Facing::DIRECTIONS {
        let rel_pos = pos.offset_i32(&dir);
        let Some(cluster) = access.cache_mut().access_cluster_mut(&pos.cluster_pos()).map(|v|v.cluster_mut()).flatten() else {
            continue;
        };
        let cluster_block_pos = rel_pos.cluster_block_pos();
        let data = cluster.raw().get(&cluster_block_pos);
        let light = data.light_state();
        let liquid_level = data.liquid_state().level();

        if !light.is_zero() {
            cluster.propagate_lighting(&cluster_block_pos);
        }

        if liquid_level > 0 {
            access.update_block(&rel_pos, |data| {
                *data.active_mut() = true;
            });
        }
    }
}

fn apply_overworld_actions(
    registry: &Arc<Registry>,
    loaded_clusters: &LoadedClusters,
    actions: &[&OverworldActionsStorage],
) {
    struct AllChanges<'a> {
        params: Vec<&'a StateChangeInfo>,
        activities: HashMap<BlockPos, bool>,
        liquids: HashMap<BlockPos, LiquidState>,
    }
    impl Default for AllChanges<'_> {
        fn default() -> Self {
            Self {
                params: Vec::with_capacity(256),
                activities: HashMap::with_capacity(256),
                liquids: HashMap::with_capacity(256),
            }
        }
    }

    let mut after_actions_by_cluster = HashMap::<ClusterPos, AllChanges>::with_capacity(4096);

    // Group actions by cluster
    for info in std::iter::empty()
        .chain(actions.iter().flat_map(|v| &v.components_infos))
        .chain(actions.iter().flat_map(|v| &v.states_infos))
    {
        let changes = after_actions_by_cluster
            .entry(info.pos.cluster_pos())
            .or_default();
        changes.params.push(info);
    }

    // Merge block activity changes
    for info in actions.iter().flat_map(|v| &v.activity_infos) {
        let changes = after_actions_by_cluster
            .entry(info.pos.cluster_pos())
            .or_default();

        let curr_activity = changes.activities.entry(info.pos).or_insert(false);
        *curr_activity |= info.active;
    }

    // Merge liquid changes
    for info in actions.iter().flat_map(|v| &v.liquid_infos) {
        let changes = after_actions_by_cluster
            .entry(info.pos.cluster_pos())
            .or_default();

        match changes.liquids.entry(info.pos) {
            hash_map::Entry::Vacant(e) => {
                e.insert(info.liquid);
            }
            hash_map::Entry::Occupied(mut e) => {
                if info.liquid.level() > e.get().level() {
                    e.insert(info.liquid);
                }
            }
        }
    }

    let all_affected_positions = Mutex::new(Vec::with_capacity(after_actions_by_cluster.len()));

    // Apply all the actions
    after_actions_by_cluster.par_iter().for_each(|(pos, changes)| {
        let mut access = OverworldAccessor::new(Arc::clone(registry), Arc::clone(loaded_clusters));
        let affected_positions: Vec<_> = changes.params.iter().map(|v| v.pos).collect();

        for action in &changes.params {
            // Safety: action's referenced data exists inside `total_after_actions`.
            // Note: `action` can't fail because we have read-write access to the cluster at `pos`.
            unsafe { action.apply(&mut access) };
        }
        for (pos, activity) in &changes.activities {
            access.update_block(pos, |v| {
                *v.active_mut() = *activity;
            });
        }
        for (pos, liquid) in &changes.liquids {
            access.set_liquid_state(pos, *liquid);
        }

        all_affected_positions.lock().push(affected_positions);
    });

    let mut access = OverworldAccessor::new(Arc::clone(registry), Arc::clone(loaded_clusters));

    // Check neighbourhood for lighting or liquid changes
    for pos in all_affected_positions.into_inner().iter().flatten() {
        check_block(registry, &mut access, pos);
    }

    // Apply lighting actions
    for info in actions.iter().flat_map(|v| &v.set_light_infos) {
        access.set_light(&info.pos, info.light);
    }
    for pos in actions.iter().flat_map(|v| &v.remove_light_positions) {
        access.remove_light(&pos);
    }
}

pub fn on_tick(
    tick: u64,
    registry: &Arc<Registry>,
    overworld_orchestrator: &mut OverworldOrchestrator,
    additional_actions: &OverworldActionsStorage,
    orchestrator_max_processing_time: Duration,
) -> OverworldUpdateResult {
    let loaded_clusters = Arc::clone(overworld_orchestrator.loaded_clusters());

    let active_blocks_actions = process_active_blocks(tick, registry, &loaded_clusters);

    // Add additional actions
    let mut actions_storages: Vec<_> = active_blocks_actions.iter().collect();
    actions_storages.push(additional_actions);

    apply_overworld_actions(registry, &loaded_clusters, &actions_storages);

    overworld_orchestrator.update(orchestrator_max_processing_time)
}
