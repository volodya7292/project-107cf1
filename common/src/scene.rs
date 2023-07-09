pub mod relation;

use crate::scene::relation::Relation;
use entity_data::{EntityId, SystemAccess};

/// Collects all `Relation`-entities in breadth-first order (parents are ordered before children).
pub fn collect_relation_tree(access: &SystemAccess, root: &EntityId) -> Vec<EntityId> {
    let relation_comps = access.component::<Relation>();
    let mut nodes = Vec::<EntityId>::with_capacity(1024);
    let mut to_visit = Vec::<EntityId>::with_capacity(nodes.capacity());

    nodes.push(*root);
    to_visit.push(*root);

    while let Some(entity) = to_visit.pop() {
        if let Some(relation) = relation_comps.get(&entity) {
            nodes.extend(&relation.children);
            // Because we're popping from the stack, insert in reverse order
            // to preserve the order of visiting children so the first is popped first.
            to_visit.extend(relation.children.iter().rev());
        }
    }

    nodes
}
