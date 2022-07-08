use crate::ecs::component::Transform;
use crate::ecs::scene_storage::Entity;
use crate::utils::IndexSet;
use std::ops::{Deref, DerefMut};

pub struct Parent(pub Entity);

#[derive(Default)]
pub struct Children {
    pub children: IndexSet<Entity>,
}

#[derive(Copy, Clone, Default)]
pub struct GlobalTransform(Transform);

impl From<Transform> for GlobalTransform {
    fn from(transform: Transform) -> Self {
        Self(transform)
    }
}

impl Deref for GlobalTransform {
    type Target = Transform;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for GlobalTransform {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
