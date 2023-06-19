pub mod execution;
pub mod main_registry;
pub mod overworld;
pub mod physics;
pub mod registry;
pub mod utils;

use crate::overworld::accessor::{
    OverworldAccessor, ReadOnlyOverworldAccessor, ReadOnlyOverworldAccessorImpl,
};
use crate::overworld::facing::Facing;
use crate::overworld::light_state::LightLevel;
use crate::overworld::liquid_state::LiquidState;
use crate::overworld::orchestrator::{OverworldOrchestrator, OverworldUpdateResult};
use crate::overworld::position::{BlockPos, ClusterPos};
use crate::overworld::raw_cluster::{BlockData, BlockDataImpl};
use crate::overworld::{ClusterState, LoadedClusters};
use crate::registry::Registry;
use common::parking_lot::Mutex;
use common::rayon::prelude::*;
use common::types::HashMap;
use common::{glm, MO_RELAXED};
use overworld::actions_storage::{OverworldActionsBuilder, OverworldActionsStorage, StateChangeInfo};
use std::collections::hash_map;
use std::sync::Arc;

pub fn on_liquid_tick(
    tick: u64,
    pos: &BlockPos,
    block_data: BlockData,
    registry: &Registry,
    access: &mut ReadOnlyOverworldAccessor,
    mut result: OverworldActionsBuilder,
) -> bool {
    if tick % 10 != 0 {
        return true;
    }
    const SPREAD_DENSITY: u8 = 2;

    let curr_liquid = *block_data.liquid_state();
    let mut new_liquid = if curr_liquid.is_source() {
        curr_liquid
    } else {
        LiquidState::new(curr_liquid.liquid_id(), 0)
    };
    let block = registry.get_block(block_data.block_id()).unwrap();

    if !block.can_pass_liquid() {
        return false;
    }

    // Check top
    {
        let rel_pos = pos.offset_i32(Facing::PositiveY.direction());
        let Some(rel_data) = access.get_block(&rel_pos) else {
            return true;
        };

        let rel_liquid = rel_data.liquid_state();

        if rel_liquid.level() > 0 && !new_liquid.is_source() {
            new_liquid = LiquidState::max(rel_liquid.liquid_id());
        }
    }

    // Check horizontal neighbours
    for dir in &Facing::XZ_DIRECTIONS {
        let rel_pos = pos.offset_i32(dir);

        let bottom_is_solid = {
            let bottom_pos = rel_pos.offset_i32(&Facing::NegativeY.direction());
            let Some(bottom_data) = access.get_block(&bottom_pos) else {
                return true;
            };
            let bottom_block = registry.get_block(bottom_data.block_id()).unwrap();
            !bottom_block.can_pass_liquid()
        };

        if !bottom_is_solid {
            continue;
        }
        let Some(rel_data) = access.get_block(&rel_pos) else {
            return true;
        };

        let rel_liquid = rel_data.liquid_state();

        if !new_liquid.is_source() && rel_liquid.level() > new_liquid.level() {
            new_liquid = LiquidState::new(
                rel_liquid.liquid_id(),
                rel_liquid.level().saturating_sub(SPREAD_DENSITY),
            );
        }
    }

    let liquid_changed = new_liquid != curr_liquid;

    if liquid_changed {
        result.set_liquid(*pos, new_liquid);
    }

    // Spread/vanish current liquid to neighbours
    {
        // Check horizontal neighbours
        for dir in &Facing::XZ_DIRECTIONS {
            let rel_pos = pos.offset_i32(&dir);
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return true;
            };
            let rel_liquid = rel_data.liquid_state();

            if liquid_changed || rel_liquid.level() < curr_liquid.level().saturating_sub(SPREAD_DENSITY) {
                result.set_activity(rel_pos, true);
            }
        }

        // Check bottom neighbour
        {
            let rel_pos = pos.offset_i32(Facing::NegativeY.direction());
            let Some(rel_data) = access.get_block(&rel_pos) else {
                return true;
            };

            let rel_block = registry.get_block(rel_data.block_id()).unwrap();
            let rel_liquid = rel_data.liquid_state();

            if liquid_changed || rel_block.can_pass_liquid() && rel_liquid.level() != curr_liquid.level() {
                result.set_activity(rel_pos, true);
            }
        }
    }

    false
}

fn on_light_tick(
    pos: &BlockPos,
    block_data: BlockData,
    registry: &Registry,
    access: &mut ReadOnlyOverworldAccessor,
    mut result: OverworldActionsBuilder,
) -> bool {
    let prev_light = block_data.light_state().components();
    let mut new_light = block_data.light_source().components();
    let block = registry.get_block(block_data.block_id()).unwrap();

    // Collect neighbouring light (vanishing)
    if block.can_pass_light() {
        for dir in Facing::DIRECTIONS {
            let rel_pos = pos.offset_i32(&dir);
            let Some(data) = access.get_block(&rel_pos) else {
                return true;
            };
            let rel_block = registry.get_block(data.block_id()).unwrap();

            let rel_light = if rel_block.can_pass_light() {
                data.light_state()
            } else {
                data.light_source()
            };

            let rel_curr_light = rel_light.components().map(|v| v.saturating_sub(1));

            new_light = new_light.sup(&rel_curr_light);
        }
    }

    // Spread current light to neighbours
    if new_light != prev_light {
        for dir in Facing::DIRECTIONS {
            let rel_pos = pos.offset_i32(&dir);
            result.set_activity(rel_pos, true);
        }

        let new_state = LightLevel::from_vec(new_light);
        result.set_light_state(*pos, new_state);
        return true;
    }

    false
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

        let Some(cluster) = o_cluster.ready() else {
            return;
        };
        let cluster = cluster.unwrap();

        let mut after_actions = OverworldActionsStorage::new();
        let mut accessor =
            OverworldAccessor::new(Arc::clone(registry), Arc::clone(loaded_clusters)).into_read_only();

        for (pos, block_data) in cluster.active_blocks() {
            let global_pos = cl_pos.to_block_pos().offset(&glm::convert(*pos.get()));
            let block = registry.get_block(block_data.block_id()).unwrap();
            let mut keep_active = false;

            // Spread/vanish liquids
            keep_active |= on_liquid_tick(
                tick,
                &global_pos,
                block_data,
                &registry,
                &mut accessor,
                after_actions.builder(),
            );

            // Spread/vanish lights
            keep_active |= on_light_tick(
                &global_pos,
                block_data,
                &registry,
                &mut accessor,
                after_actions.builder(),
            );

            // Block-specific on-tick event
            if let Some(on_tick) = &block.event_handlers().on_tick {
                keep_active |= on_tick(
                    tick,
                    &global_pos,
                    block_data,
                    &registry,
                    &mut accessor,
                    after_actions.builder(),
                );
            }

            after_actions.set_active(global_pos, keep_active);
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
        access.update_block(pos, |data| {
            *data.light_state_mut() = LightLevel::ZERO;
        });
    }
    if !curr_block.can_pass_liquid() {
        access.update_block(pos, |data| {
            *data.liquid_state_mut() = LiquidState::NONE;
        });
    }

    for dir in &Facing::DIRECTIONS {
        let rel_pos = pos.offset_i32(&dir);
        access.update_block(&rel_pos, |data| {
            *data.active_mut() = true;
        });
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
        lights_sources: HashMap<BlockPos, LightLevel>,
        lights_states: HashMap<BlockPos, LightLevel>,
    }
    impl Default for AllChanges<'_> {
        fn default() -> Self {
            Self {
                params: Vec::with_capacity(256),
                activities: HashMap::with_capacity(256),
                liquids: HashMap::with_capacity(256),
                lights_sources: HashMap::with_capacity(256),
                lights_states: HashMap::with_capacity(256),
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

    // Merge light source changes
    for info in actions.iter().flat_map(|v| &v.light_source_infos) {
        let changes = after_actions_by_cluster
            .entry(info.pos.cluster_pos())
            .or_default();

        if let hash_map::Entry::Vacant(e) = changes.lights_sources.entry(info.pos) {
            e.insert(info.light);
        }
    }

    // Merge light state changes
    for info in actions.iter().flat_map(|v| &v.light_state_infos) {
        let changes = after_actions_by_cluster
            .entry(info.pos.cluster_pos())
            .or_default();

        if let hash_map::Entry::Vacant(e) = changes.lights_states.entry(info.pos) {
            e.insert(info.light);
        }
    }

    let all_affected_positions = Mutex::new(Vec::with_capacity(after_actions_by_cluster.len()));

    // Apply all the actions
    after_actions_by_cluster.par_iter().for_each(|(_, changes)| {
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
        for (pos, light) in &changes.lights_sources {
            access.set_light_source(pos, *light);
        }
        for (pos, light) in &changes.lights_states {
            access.set_light_state(pos, *light);
        }

        all_affected_positions.lock().push(affected_positions);
    });

    let mut access = OverworldAccessor::new(Arc::clone(registry), Arc::clone(loaded_clusters));

    // Check neighbourhood for lighting or liquid changes
    for pos in all_affected_positions.into_inner().iter().flatten() {
        check_block(registry, &mut access, pos);
    }
}

pub fn on_tick(
    tick: u64,
    registry: &Arc<Registry>,
    overworld_orchestrator: &mut OverworldOrchestrator,
    additional_actions: &OverworldActionsStorage,
) -> OverworldUpdateResult {
    let loaded_clusters = Arc::clone(overworld_orchestrator.loaded_clusters());

    let active_blocks_actions = process_active_blocks(tick, registry, &loaded_clusters);

    // Add additional actions
    let mut actions_storages: Vec<_> = active_blocks_actions.iter().collect();
    actions_storages.push(additional_actions);

    apply_overworld_actions(registry, &loaded_clusters, &actions_storages);

    overworld_orchestrator.update()
}
