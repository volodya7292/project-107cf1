pub mod relation;

use crate::scene::relation::Relation;
use entity_data::{EntityId, SystemAccess};

/// Walks all `Relation`-entities in depth-first order.
/// Implements stack management using parent info `PI` for each visit.
/// `F` returns `PI` that is be fed info `F` when visiting its children.
pub fn walk_relation_tree<PI: Clone, F: FnMut(&EntityId, PI) -> PI>(
    access: &SystemAccess,
    root: &EntityId,
    root_parent_info: PI,
    mut f: F,
) {
    let relation_comps = access.component::<Relation>();
    let mut to_visit = Vec::<(EntityId, PI)>::with_capacity(1024);
    to_visit.push((*root, root_parent_info));

    while let Some((entity, entry_info)) = to_visit.pop() {
        let parent_info = f(&entity, entry_info);

        if let Some(relation) = relation_comps.get(&entity) {
            // Because we're popping from the stack, insert in reverse order
            // to preserve the order of visiting children so the first is popped first.
            to_visit.extend(relation.children.iter().rev().map(|e| (*e, parent_info.clone())));
        }
    }
}

/// Collects all `Relation`-entities in depth-first order.
pub fn collect_relation_tree_out(access: &SystemAccess, root: &EntityId, out_nodes: &mut Vec<EntityId>) {
    let relation_comps = access.component::<Relation>();
    let mut to_visit = Vec::<EntityId>::with_capacity(out_nodes.capacity());

    to_visit.push(*root);

    while let Some(entity) = to_visit.pop() {
        out_nodes.push(entity);

        if let Some(relation) = relation_comps.get(&entity) {
            // Because we're popping from the stack, insert in reverse order
            // to preserve the order of visiting children so the first is popped first.
            to_visit.extend(relation.children.iter().rev());
        }
    }
}

/// Collects all `Relation`-entities in depth-first order.
pub fn collect_relation_tree(access: &SystemAccess, root: &EntityId) -> Vec<EntityId> {
    let mut out = Vec::with_capacity(1024);
    collect_relation_tree_out(access, root, &mut out);
    out
}
