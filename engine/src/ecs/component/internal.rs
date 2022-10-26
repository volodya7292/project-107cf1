use std::ops::{Deref, DerefMut};

use entity_data::EntityId;

use crate::ecs::component::Transform;
use crate::utils::IndexSet;

#[derive(Default)]
pub struct Relation {
    pub parent: EntityId,
    pub children: IndexSet<EntityId>,
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
