use crate::ecs::component::TransformC;
use common::glm;
use common::nalgebra::Rotation3;
use glm::{DVec3, Mat4, Vec3};

#[derive(Copy, Clone)]
pub struct HierarchyCacheC {
    pub(crate) active: bool,
    pub(crate) position: DVec3,
    pub(crate) rotation: Vec3,
    pub(crate) scale: Vec3,
}

impl Default for HierarchyCacheC {
    fn default() -> Self {
        Self {
            active: true,
            position: Default::default(),
            rotation: Default::default(),
            scale: Vec3::from_element(1.0),
        }
    }
}

impl HierarchyCacheC {
    /// Applies `parent` transformation to `child_transform` and sets the result.
    pub fn set_transform(&mut self, parent: &Self, child_transform: &TransformC) {
        if child_transform.use_parent_transform {
            self.position = parent.position + child_transform.position;
            self.rotation =
                (parent.rotation + child_transform.rotation).map(|v| v % (std::f32::consts::PI * 2.0));
            self.scale = parent.scale.component_mul(&child_transform.scale);
        } else {
            self.position = child_transform.position;
            self.rotation = child_transform.rotation;
            self.scale = child_transform.scale;
        }
    }

    pub fn position_f32(&self) -> Vec3 {
        glm::convert(self.position)
    }

    pub fn transform_matrix_f32(&self) -> Mat4 {
        let mut mat = Mat4::new_nonuniform_scaling(&self.scale);
        mat = Rotation3::from_axis_angle(&Vec3::x_axis(), self.rotation.x).to_homogeneous() * mat;
        mat = Rotation3::from_axis_angle(&Vec3::y_axis(), self.rotation.y).to_homogeneous() * mat;
        mat = Rotation3::from_axis_angle(&Vec3::z_axis(), self.rotation.z).to_homogeneous() * mat;
        mat = Mat4::new_translation(&glm::convert(self.position)) * mat;
        mat
    }
}
