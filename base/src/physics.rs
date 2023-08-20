pub mod aabb;

use crate::overworld::accessor::ReadOnlyOverworldAccessorImpl;
use crate::overworld::position::BlockPos;
use crate::overworld::raw_cluster::BlockDataImpl;
use crate::overworld::Overworld;
use aabb::AABB;
use approx::AbsDiffEq;
use common::glm;
use common::glm::{DVec3, I64Vec3, Vec3};

pub const MOTION_EPSILON: f64 = 1e-10;
pub const G_ACCEL: f64 = 12.0;

pub fn calc_force(mass: f32, accel: Vec3) -> Vec3 {
    accel * mass
}

pub fn calc_acceleration(forces: Vec3, object_mass: f32) -> Vec3 {
    forces / object_mass
}

pub fn collision_delta_many2one(aabbs: &[AABB], other: &AABB, resolve_direction: &DVec3) -> DVec3 {
    if aabbs.is_empty() {
        return DVec3::zeros();
    }

    let deltas: Vec<_> = aabbs
        .iter()
        .map(|aabb| aabb.collision_delta(other, resolve_direction))
        .collect();

    let positive_delta = deltas
        .iter()
        .map(|v| v.sup(&DVec3::zeros()))
        .fold(DVec3::zeros(), |total_delta, delta| total_delta.sup(&delta));

    let negative_delta = deltas
        .iter()
        .map(|v| v.inf(&DVec3::zeros()))
        .fold(DVec3::zeros(), |total_delta, delta| total_delta.inf(&delta));

    positive_delta + negative_delta
}

impl Overworld {
    /// If collisions can't be resolved, applies resolutions for each collider
    /// effectively averaging all resolutions. Returns new object position.
    /// This resolution may be unstable (the object at new position may collide with other objects).
    pub fn resolve_collisions_fairly(
        &self,
        curr_object_pos: DVec3,
        motion_delta: DVec3,
        object_aabb: &AABB,
    ) -> DVec3 {
        let aabb_in_full_motion = object_aabb.combine(&object_aabb.translate(&motion_delta));
        let global_object_aabb = object_aabb.translate(&curr_object_pos);
        let global_motion_aabb = aabb_in_full_motion.translate(&curr_object_pos);

        // Use motion_aabb to account for extreme motion deltas to prevent 'tunneling'
        let start: I64Vec3 = glm::convert_unchecked(glm::floor(global_motion_aabb.min()));
        let size: I64Vec3 = glm::convert_unchecked(glm::ceil(&global_motion_aabb.size()).add_scalar(1.0));

        let reg = self.main_registry().registry();
        let mut access = self.access();
        let mut blocks_aabbs = Vec::<AABB>::with_capacity((size.x * size.y * size.z) as usize * 2);

        for x in 0..size.x {
            for y in 0..size.y {
                for z in 0..size.z {
                    let pos = start + I64Vec3::new(x, y, z);

                    let Some(entry) = access.get_block(&BlockPos(pos)) else {
                        continue;
                    };

                    let block_id = entry.block_id();
                    let block = reg.get_block(block_id).unwrap();

                    if let Some(model) = reg.get_block_model(block.model_id()) {
                        let pos: DVec3 = glm::convert(pos);
                        blocks_aabbs.extend(model.aabbs().iter().map(|v| v.translate(&pos)));
                    }
                }
            }
        }

        let res_dir = -motion_delta;
        let mut new_object_pos = curr_object_pos;

        let x_delta = DVec3::new(motion_delta.x, 0.0, 0.0);
        let y_delta = DVec3::new(0.0, motion_delta.y, 0.0);
        let z_delta = DVec3::new(0.0, 0.0, motion_delta.z);

        // Resolve collisions on separate axes independently of each other
        // to correctly calculate collision deltas

        new_object_pos += x_delta;
        let in_motion_aabb = global_object_aabb.combine(&object_aabb.translate(&new_object_pos));
        new_object_pos.x -= collision_delta_many2one(&blocks_aabbs, &in_motion_aabb, &res_dir).x;

        new_object_pos += y_delta;
        let in_motion_aabb = global_object_aabb.combine(&object_aabb.translate(&new_object_pos));
        new_object_pos.y -= collision_delta_many2one(&blocks_aabbs, &in_motion_aabb, &res_dir).y;

        new_object_pos += z_delta;
        let in_motion_aabb = global_object_aabb.combine(&object_aabb.translate(&new_object_pos));
        new_object_pos.z -= collision_delta_many2one(&blocks_aabbs, &in_motion_aabb, &res_dir).z;

        new_object_pos
    }

    /// If collisions can't be resolved, return the initial position
    /// without applying the motion delta. Returns new object position.
    /// Ensures that the object can't move in unresolvable situations.
    pub fn try_resolve_collisions(
        &self,
        curr_object_pos: DVec3,
        motion_delta: DVec3,
        object_aabb: &AABB,
    ) -> DVec3 {
        let new_pos = self.resolve_collisions_fairly(curr_object_pos, motion_delta, object_aabb);
        // Use another fair collision resolution to check the first resolution stability.
        let new_pos2 = self.resolve_collisions_fairly(new_pos, DVec3::zeros(), object_aabb);

        if new_pos.abs_diff_eq(&new_pos2, MOTION_EPSILON) {
            new_pos
        } else {
            // The collision can't be resolved
            curr_object_pos
        }
    }
}
