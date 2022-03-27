use crate::component::Transform;
use std::ops::Deref;

#[derive(Copy, Clone, Default)]
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
