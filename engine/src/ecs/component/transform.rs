use nalgebra::Rotation3;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, Mat4, Vec3};

#[derive(Copy, Clone)]
pub struct Transform {
    // Note: use DVec3 to support large distances
    pub(crate) position: DVec3,
    pub(crate) rotation: Vec3,
    pub(crate) scale: Vec3,
}

impl Transform {
    pub fn new(position: DVec3, rotation: Vec3, scale: Vec3) -> Transform {
        Transform {
            position,
            rotation,
            scale,
        }
    }

    /// Applies `self` transformation to `other`
    pub fn combine(&self, other: &Transform) -> Transform {
        Transform {
            position: self.position + other.position,
            rotation: (self.rotation + other.rotation).map(|v| v % (std::f32::consts::PI * 2.0)),
            scale: self.scale.component_mul(&other.scale),
        }
    }

    pub fn matrix_f32(&self) -> Mat4 {
        let mut mat = Mat4::identity();
        mat.prepend_translation_mut(&glm::convert(self.position));
        mat *=
            Rotation3::from_euler_angles(self.rotation.x, self.rotation.y, self.rotation.z).to_homogeneous();
        mat.prepend_nonuniform_scaling_mut(&self.scale);
        mat
    }

    pub fn position(&self) -> &DVec3 {
        &self.position
    }

    pub fn position_f32(&self) -> Vec3 {
        glm::convert(self.position)
    }

    pub fn rotation(&self) -> &Vec3 {
        &self.rotation
    }

    pub fn scale(&self) -> &Vec3 {
        &self.scale
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: DVec3::new(0.0, 0.0, 0.0),
            rotation: Vec3::new(0.0, 0.0, 0.0),
            scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }
}
