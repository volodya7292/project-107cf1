pub mod aabb;

use crate::game::overworld::cluster::BlockDataImpl;
use crate::game::overworld::Overworld;
use aabb::AABB;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3};

pub const MOTION_EPSILON: f64 = 1e-10;
pub const G_ACCEL: f64 = 14.0;

fn combine_collision_deltas(a: DVec3, b: DVec3) -> DVec3 {
    a.zip_map(&b, |a, b| {
        if a >= 0.0 && b >= 0.0 {
            a.max(b)
        } else if a <= 0.0 && b <= 0.0 {
            a.min(b)
        } else {
            0.0
        }
    })
}

pub fn collision_delta_many2one(aabbs: &[AABB], other: &AABB, resolve_direction: &DVec3) -> DVec3 {
    aabbs.iter().fold(DVec3::default(), |acc, v| {
        combine_collision_deltas(acc, v.collision_delta(other, resolve_direction))
    })
}

impl Overworld {
    /// Returns new object position
    pub fn move_entity(&self, curr_object_pos: DVec3, motion_delta: DVec3, object_aabb: &AABB) -> DVec3 {
        let aabb_in_full_motion = object_aabb.combine(&object_aabb.translate(motion_delta));
        let global_object_aabb = object_aabb.translate(curr_object_pos);
        let global_motion_aabb = aabb_in_full_motion.translate(curr_object_pos);

        // Use motion_aabb to account for extreme motion deltas to prevent 'tunneling'
        let start: I64Vec3 = glm::convert_unchecked(glm::floor(&global_motion_aabb.min()));
        let size: I64Vec3 = glm::convert_unchecked(glm::ceil(&global_motion_aabb.size()).add_scalar(1.0));

        let reg = self.main_registry().registry();
        let mut access = self.access();
        let mut blocks_aabbs = Vec::<AABB>::with_capacity((size.x * size.y * size.z) as usize * 2);

        for x in 0..size.x {
            for y in 0..size.y {
                for z in 0..size.z {
                    let pos = start + I64Vec3::new(x, y, z);

                    if let Some(entry) = access.get_block(&pos) {
                        let block_id = entry.block_id();
                        let block = reg.get_block(block_id).unwrap();
                        if !block.has_textured_model() {
                            continue;
                        }
                        let model = reg.get_textured_block_model(block.textured_model()).unwrap();

                        let pos: DVec3 = glm::convert(pos);
                        blocks_aabbs.extend(model.aabbs().iter().map(|v| v.translate(pos)));
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
        let in_motion_aabb = global_object_aabb.combine(&object_aabb.translate(new_object_pos));
        new_object_pos.x -= collision_delta_many2one(&blocks_aabbs, &in_motion_aabb, &res_dir).x;

        new_object_pos += y_delta;
        let in_motion_aabb = global_object_aabb.combine(&object_aabb.translate(new_object_pos));
        new_object_pos.y -= collision_delta_many2one(&blocks_aabbs, &in_motion_aabb, &res_dir).y;

        new_object_pos += z_delta;
        let in_motion_aabb = global_object_aabb.combine(&object_aabb.translate(new_object_pos));
        new_object_pos.z -= collision_delta_many2one(&blocks_aabbs, &in_motion_aabb, &res_dir).z;

        new_object_pos
    }
}
