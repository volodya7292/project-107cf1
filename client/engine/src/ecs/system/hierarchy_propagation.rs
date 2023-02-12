use crate::ecs::component::internal::GlobalTransformC;
use crate::ecs::component::TransformC;
use common::scene::relation::Relation;
use common::types::HashSet;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use std::time::Instant;

// Propagates transform hierarchy and calculates global transforms
pub(crate) struct HierarchyPropagation<'a> {
    pub root_entity: EntityId,
    pub dirty_transform_comps: HashSet<EntityId>,
    pub ordered_entities: &'a mut Vec<EntityId>,
    pub changed_global_transforms: Vec<EntityId>,
    pub run_time: f64,
}

struct StackEntry {
    entity: EntityId,
    parent_global_transform_changed: bool,
    parent_global_transform: GlobalTransformC,
}

impl SystemHandler for HierarchyPropagation<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let relation_comps = data.component::<Relation>();
        let transform_comps = data.component::<TransformC>();
        let mut global_transform_comps = data.component_mut::<GlobalTransformC>();
        let mut stack = Vec::<StackEntry>::with_capacity(transform_comps.count_entities());

        stack.push(StackEntry {
            entity: self.root_entity,
            parent_global_transform_changed: false,
            parent_global_transform: Default::default(),
        });

        self.ordered_entities.clear();

        // Recursion using loop
        while let Some(StackEntry {
            entity,
            parent_global_transform_changed: parent_transform_changed,
            parent_global_transform,
        }) = stack.pop()
        {
            self.ordered_entities.push(entity);

            let global_transform_changed =
                parent_transform_changed || self.dirty_transform_comps.contains(&entity);

            let global_transform = if let Some(global_transform) = global_transform_comps.get_mut(&entity) {
                if global_transform_changed {
                    let model_transform = transform_comps.get(&entity).unwrap();
                    *global_transform = parent_global_transform.combine(model_transform);
                    self.changed_global_transforms.push(entity);
                }
                *global_transform
            } else {
                Default::default()
            };

            if let Some(relation) = relation_comps.get(&entity) {
                // Because we're popping from the stack, insert in reverse order
                // to preserve the right order of insertion to `ordered_entities`
                stack.extend(relation.children.iter().rev().map(|e| StackEntry {
                    entity: *e,
                    parent_global_transform_changed: global_transform_changed,
                    parent_global_transform: global_transform,
                }));
            }
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
