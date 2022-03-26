use crate::component::Transform;
use nalgebra as na;
use std::ops::Deref;

#[derive(Copy, Clone)]
pub struct WorldTransform(Transform);

impl From<Transform> for WorldTransform {
    fn from(transform: Transform) -> Self {
        Self(transform)
    }
}

impl Deref for WorldTransform {
    type Target = Transform;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
