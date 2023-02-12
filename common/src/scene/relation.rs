use crate::types::IndexSet;
use entity_data::EntityId;

#[derive(Default)]
pub struct Relation {
    pub parent: EntityId,
    pub children: IndexSet<EntityId>,
}
