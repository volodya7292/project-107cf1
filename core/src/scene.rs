use entity_data::{EntityId, SystemAccess};

use crate::scene::relation::Relation;

pub mod relation;

pub fn collect_children_recursively(access: &SystemAccess, entity: EntityId, children: &mut Vec<EntityId>) {
    let relation_comps = access.component::<Relation>();

    let mut stack = Vec::<EntityId>::with_capacity(64);
    stack.push(entity);

    while let Some(entity) = stack.pop() {
        if let Some(relation) = relation_comps.get(&entity) {
            children.extend(&relation.children);
            stack.extend(&relation.children);
        }
    }
}
