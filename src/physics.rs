use crate::game::overworld::cluster::{BlockDataImpl, Cluster};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, U32Vec3};

pub struct AABB {
    min: DVec3,
    max: DVec3,
}

impl AABB {
    pub fn size(&self) -> DVec3 {
        self.max - self.min
    }

    pub fn collision_delta(&self, other: &AABB) -> DVec3 {
        let mut delta = DVec3::from_element(0.0);

        for i in 0..3 {
            if (other.min[i] >= self.min[i] && other.max[i] <= self.max[i])
                || (other.min[i] <= self.min[i] && other.max[i] >= self.max[i])
            {
                continue;
            }
            if other.min >= self.min {
                delta[i] = (other.min[i] - self.max[i]).min(0.0);
            } else {
                delta[i] = (other.max[i] - self.min[i]).max(0.0);
            }
        }

        delta
    }
}

impl Cluster {
    pub fn calc_collision_delta(&self, motion_delta: DVec3, aabb: &AABB) -> DVec3 {
        let size: I64Vec3 = glm::try_convert(glm::ceil(&aabb.size()).add_scalar(1.0)).unwrap();
        let mut error = DVec3::new(0.0, 0.0, 0.0);

        for x in 0..size.x {
            for y in 0..size.y {
                for z in 0..size.z {
                    let pos: U32Vec3 = todo!();
                    let entry = self.get(pos);
                    let block = entry.block();
                }
            }
        }

        error
    }
}
