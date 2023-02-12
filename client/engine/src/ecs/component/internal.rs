use crate::ecs::component::TransformC;
use common::glm;
use common::nalgebra::Rotation3;
use glm::{DVec3, Mat4, Vec3};

#[derive(Copy, Clone)]
pub struct GlobalTransformC {
    pub(crate) position: DVec3,
    pub(crate) rotation: Vec3,
    pub(crate) scale: Vec3,
}

impl Default for GlobalTransformC {
    fn default() -> Self {
        Self {
            position: Default::default(),
            rotation: Default::default(),
            scale: Vec3::from_element(1.0),
        }
    }
}

impl GlobalTransformC {
    /// Applies `self` transformation to `transform` and returns the result.
    pub fn combine(&self, transform: &TransformC) -> GlobalTransformC {
        if transform.use_parent_transform {
            GlobalTransformC {
                position: self.position + transform.position,
                rotation: (self.rotation + transform.rotation).map(|v| v % (std::f32::consts::PI * 2.0)),
                scale: self.scale.component_mul(&transform.scale),
            }
        } else {
            GlobalTransformC {
                position: transform.position,
                rotation: transform.rotation,
                scale: transform.scale,
            }
        }
    }

    pub fn position_f32(&self) -> Vec3 {
        glm::convert(self.position)
    }

    pub fn matrix_f32(&self) -> Mat4 {
        let mut mat = Mat4::new_nonuniform_scaling(&self.scale);
        mat = Rotation3::from_axis_angle(&Vec3::x_axis(), self.rotation.x).to_homogeneous() * mat;
        mat = Rotation3::from_axis_angle(&Vec3::y_axis(), self.rotation.y).to_homogeneous() * mat;
        mat = Rotation3::from_axis_angle(&Vec3::z_axis(), self.rotation.z).to_homogeneous() * mat;
        mat = Mat4::new_translation(&glm::convert(self.position)) * mat;
        mat
    }
}
