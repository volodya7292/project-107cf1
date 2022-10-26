use entity_data::EntityId;

use engine::utils::IndexSet;

#[derive(Default)]
pub struct Relation {
    pub parent: EntityId,
    pub children: IndexSet<EntityId>,
}
