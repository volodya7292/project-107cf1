use crate::renderer::component::Transform;
use nalgebra as na;

#[derive(Copy, Clone)]
pub struct ModelTransform {
    pub(in crate::renderer) matrix: na::Matrix4<f32>,
    pub(in crate::renderer) position: na::Vector3<f32>,
    pub(in crate::renderer) scale: na::Vector3<f32>,
    pub(in crate::renderer) changed: bool,
}

impl ModelTransform {
    pub fn from_transform(transform: &Transform) -> ModelTransform {
        ModelTransform {
            matrix: transform.matrix(),
            position: transform.position,
            scale: transform.scale,
            changed: true,
        }
    }
}

impl Default for ModelTransform {
    fn default() -> Self {
        ModelTransform {
            matrix: na::Matrix4::identity(),
            position: Default::default(),
            scale: na::Vector3::from_element(1.0),
            changed: true,
        }
    }
}
