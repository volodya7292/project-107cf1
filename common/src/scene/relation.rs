use crate::types::IndexSet;
use entity_data::EntityId;

pub struct Relation {
    pub active: bool,
    pub parent: EntityId,
    pub children: IndexSet<EntityId>,
}

impl Default for Relation {
    fn default() -> Self {
        Self {
            active: true,
            parent: Default::default(),
            children: Default::default(),
        }
    }
}
