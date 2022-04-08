use crate::game::overworld::block_component::Facing;
use crate::physics::MOTION_EPSILON;
use nalgebra_glm::{DVec3, I32Vec3};

#[derive(Debug, Copy, Clone)]
pub struct AABB {
    min: DVec3,
    max: DVec3,
}

#[derive(Debug, Copy, Clone)]
pub struct AABBRayIntersection {
    facing: Facing,
    point: DVec3,
}

impl AABBRayIntersection {
    pub fn facing(&self) -> Facing {
        self.facing
    }

    pub fn point(&self) -> &DVec3 {
        &self.point
    }
}

impl AABB {
    pub fn new(min: DVec3, max: DVec3) -> Self {
        Self { min, max }
    }

    /// Creates a new AABB centered at origin of specified size.
    pub fn from_size(size: DVec3) -> Self {
        let half = size * 0.5;
        Self {
            min: -half,
            max: half,
        }
    }

    pub fn min(&self) -> &DVec3 {
        &self.min
    }

    pub fn max(&self) -> &DVec3 {
        &self.max
    }

    pub fn center(&self) -> DVec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> DVec3 {
        self.max - self.min
    }

    pub fn translate(&self, translation: DVec3) -> Self {
        Self {
            min: self.min + translation,
            max: self.max + translation,
        }
    }

    pub fn combine(&self, other: &AABB) -> Self {
        Self {
            min: self.min.inf(&other.min),
            max: self.max.sup(&other.max),
        }
    }

    pub fn collides_with(&self, other: &Self) -> bool {
        self.min < other.max && self.max > other.min
    }

    /// Calculates delta of collision of `other` to `self`
    pub fn collision_delta(&self, other: &Self, resolve_direction: &DVec3) -> DVec3 {
        let mut delta = DVec3::from_element(0.0);

        if !self.collides_with(other) {
            return DVec3::default();
        }

        for i in 0..3 {
            // `other` is inside `self`
            if other.min[i] >= self.min[i] && other.max[i] <= self.max[i] {
                continue;
            }
            // `self` is inside `other`
            if other.min[i] <= self.min[i] && other.max[i] >= self.max[i] {
                if resolve_direction[i] > MOTION_EPSILON {
                    delta[i] = (other.min[i] - self.max[i] - MOTION_EPSILON).min(0.0);
                } else if resolve_direction[i] < -MOTION_EPSILON {
                    delta[i] = (other.max[i] - self.min[i] + MOTION_EPSILON).max(0.0);
                } else {
                    continue;
                }
            }

            // Note: add MOTION_EPSILON to collision delta to account for f64 precision errors
            if other.min[i] >= self.min[i] {
                delta[i] = (other.min[i] - self.max[i] - MOTION_EPSILON).min(0.0);
            } else {
                delta[i] = (other.max[i] - self.min[i] + MOTION_EPSILON).max(0.0);
            }
        }

        delta
    }

    /// If the ray intersects `self`, returns intersection facing and point.
    pub fn ray_intersection(&self, ray_origin: &DVec3, ray_dir: &DVec3) -> Option<AABBRayIntersection> {
        let t0 = (self.min - ray_origin).component_div(ray_dir);
        let t1 = (self.max - ray_origin).component_div(ray_dir);
        let t_min = t0.inf(&t1).max();
        let t_max = t0.sup(&t1).min();

        if t_min <= t_max {
            let inter = ray_origin
                + if t_min < 0.0 {
                    DVec3::default()
                } else {
                    ray_dir * t_min
                };

            let inter_relative = inter - self.center();
            let facing_comp = inter_relative.iamax();

            let mut facing_dir = I32Vec3::default();
            facing_dir[facing_comp] = inter_relative[facing_comp].signum() as i32;

            let facing = Facing::from_direction(facing_dir).unwrap();

            Some(AABBRayIntersection { facing, point: inter })
        } else {
            None
        }
    }
}
