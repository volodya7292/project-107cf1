use nalgebra::{Matrix4, Rotation3, Vector3};

pub struct Transform {
    pub(in crate::render_engine) position: Vector3<f32>,
    pub(in crate::render_engine) rotation: Vector3<f32>,
    pub(in crate::render_engine) scale: Vector3<f32>,
}

impl Transform {
    pub fn new(position: Vector3<f32>, rotation: Vector3<f32>, scale: Vector3<f32>) -> Transform {
        Transform {
            position,
            rotation,
            scale,
        }
    }

    pub fn matrix(&self) -> Matrix4<f32> {
        let mut mat = Matrix4::<f32>::identity();
        mat.prepend_translation_mut(&self.position);
        mat *=
            Rotation3::from_euler_angles(self.rotation.x, self.rotation.y, self.rotation.z).to_homogeneous();
        mat.prepend_nonuniform_scaling_mut(&self.scale);
        mat
    }

    pub fn position(&self) -> &Vector3<f32> {
        &self.position
    }

    pub fn rotation(&self) -> &Vector3<f32> {
        &self.rotation
    }

    pub fn scale(&self) -> &Vector3<f32> {
        &self.scale
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 0.0, 0.0),
            rotation: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
        }
    }
}
