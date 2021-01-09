use crate::renderer::component::model_transform::ModelTransform;
use nalgebra as na;

#[derive(Copy, Clone)]
pub struct WorldTransform {
    pub(in crate::renderer) matrix: na::Matrix4<f32>,
    pub(in crate::renderer) position: na::Vector3<f32>,
    pub(in crate::renderer) scale: na::Vector3<f32>,
}

impl WorldTransform {
    pub fn from_model_transform(model_transform: &ModelTransform) -> WorldTransform {
        WorldTransform {
            matrix: model_transform.matrix,
            position: model_transform.position,
            scale: model_transform.scale,
        }
    }

    pub fn combine(&self, model_transform: &ModelTransform) -> WorldTransform {
        WorldTransform {
            matrix: self.matrix * model_transform.matrix,
            position: self.position + model_transform.position,
            scale: self.scale.component_mul(&model_transform.scale),
        }
    }
}

impl Default for WorldTransform {
    fn default() -> Self {
        WorldTransform {
            matrix: na::Matrix4::identity(),
            position: Default::default(),
            scale: na::Vector3::from_element(1.0),
        }
    }
}
