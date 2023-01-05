use entity_data::EntityId;

use crate::utils::IndexSet;

#[derive(Default)]
pub struct Relation {
    pub parent: EntityId,
    pub children: IndexSet<EntityId>,
}
