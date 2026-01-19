use crate::types::IndexSet;
use entity_data::EntityId;
use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Eq)]
struct ChildEntityId {
    value: EntityId,
    /// Specifies element's order among its siblings. The order is ascending.
    order: Option<u32>,
}

impl Hash for ChildEntityId {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl PartialEq<Self> for ChildEntityId {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

pub struct Relation {
    pub active: bool,
    pub parent: EntityId,
    children: IndexSet<ChildEntityId>,
}

impl Relation {
    pub fn add_children(&mut self, children: impl Iterator<Item = EntityId>) {
        self.children.extend(children.map(|entity| ChildEntityId {
            value: entity,
            order: None,
        }));
    }

    /// Faster than [Self::ordered_children].
    pub fn unordered_children(&self) -> impl DoubleEndedIterator<Item = EntityId> + '_ {
        self.children.iter().map(|v| v.value)
    }

    /// Slower than [Self::unordered_children].
    pub fn ordered_children(&self) -> impl DoubleEndedIterator<Item = EntityId> {
        self.children
            .clone()
            .sorted_unstable_by(|a, b| a.order.cmp(&b.order))
            .map(|v| v.value)
    }

    pub fn remove_child(&mut self, entity: &EntityId) {
        // Ignore order when removing because we have self.ordered_children()
        self.children.swap_remove(&ChildEntityId {
            value: *entity,
            order: None,
        });
    }

    pub fn set_child_order(&mut self, entity: &EntityId, order: Option<u32>) {
        let was_present = self
            .children
            .replace(ChildEntityId {
                value: *entity,
                order,
            })
            .is_some();
        assert!(was_present);
    }

    pub fn clear_children(&mut self) {
        self.children.clear();
    }

    pub fn num_children(&self) -> usize {
        self.children.len()
    }
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
