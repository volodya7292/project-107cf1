use entity_data::{EntityId, EntityStorage, SystemAccess};

use crate::scene::relation::Relation;

pub mod relation;

/// Pushes new children to `children` in breadth-first order.
pub fn collect_children_recursively(access: &SystemAccess, entity: &EntityId, children: &mut Vec<EntityId>) {
    let relation_comps = access.component::<Relation>();

    let mut to_visit = Vec::<EntityId>::with_capacity(256);
    to_visit.push(*entity);

    while let Some(entity) = to_visit.pop() {
        if let Some(relation) = relation_comps.get(&entity) {
            children.extend(&relation.children);
            to_visit.extend(&relation.children);
        }
    }
}
