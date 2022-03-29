use crate::ecs::component::Transform;
use crate::ecs::scene::Entity;
use crate::utils::IndexSet;
use std::ops::Deref;

pub struct Parent(pub Entity);

#[derive(Default)]
pub struct Children {
    pub children: IndexSet<Entity>,
}

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
