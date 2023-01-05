use std::time::Instant;

use entity_data::{EntityId, SystemAccess, SystemHandler};

use core::utils::HashSet;

use crate::ecs::component;
use crate::ecs::component::internal::{GlobalTransform, Relation};

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
    parent_global_transform: GlobalTransform,
}

impl SystemHandler for HierarchyPropagation<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let relation_comps = data.component::<Relation>();
        let transform_comps = data.component::<component::Transform>();
        let mut global_transform_comps = data.component_mut::<GlobalTransform>();
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

            let global_transform = global_transform_comps.get_mut(&entity).unwrap();

            if global_transform_changed {
                let model_transform = transform_comps.get(&entity).unwrap();

                let new_global_transform: GlobalTransform =
                    parent_global_transform.combine(model_transform).into();

                *global_transform = new_global_transform;
                self.changed_global_transforms.push(entity);
            }

            if let Some(children) = relation_comps.get(&entity).map(|v| &v.children) {
                // Because we're popping from the stack, insert in reverse order
                // to preserve the right order of insertion to `ordered_entities`
                stack.extend(children.iter().rev().map(|e| StackEntry {
                    entity: *e,
                    parent_global_transform_changed: global_transform_changed,
                    parent_global_transform: *global_transform,
                }));
            }
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
