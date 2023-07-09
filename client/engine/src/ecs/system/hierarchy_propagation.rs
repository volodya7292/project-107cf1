use crate::ecs::component::internal::HierarchyCacheC;
use crate::ecs::component::TransformC;
use common::scene;
use common::scene::relation::Relation;
use common::types::HashSet;
use entity_data::{EntityId, SystemAccess, SystemHandler};
use std::time::Instant;

// Propagates transform hierarchy and calculates global transforms
pub(crate) struct HierarchyPropagation<'a> {
    pub root_entity: EntityId,
    pub dirty_relations: HashSet<EntityId>,
    pub dirty_transforms: HashSet<EntityId>,
    pub ordered_entities: &'a mut Vec<EntityId>,
    pub changed_h_caches: Vec<EntityId>,
    pub run_time: f64,
}

struct StackEntry {
    entity: EntityId,
    parent_global_transform_changed: bool,
    parent_h_cache: HierarchyCacheC,
}

impl SystemHandler for HierarchyPropagation<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let relation_comps = data.component::<Relation>();
        let transform_comps = data.component::<TransformC>();
        let mut h_cache_comps = data.component_mut::<HierarchyCacheC>();
        let mut stack = Vec::<StackEntry>::with_capacity(transform_comps.count_entities());

        stack.push(StackEntry {
            entity: self.root_entity,
            // parent_active: relation_comps.get(&self.root_entity).unwrap().active,
            parent_global_transform_changed: false,
            parent_h_cache: Default::default(),
        });

        self.ordered_entities.clear();

        // Recursion using loop from top to bottom
        while let Some(StackEntry {
            entity,
            parent_global_transform_changed: parent_transform_changed,
            parent_h_cache,
        }) = stack.pop()
        {
            self.ordered_entities.push(entity);

            let global_transform_changed =
                parent_transform_changed || self.dirty_transforms.contains(&entity);

            let global_h_cache = if let Some(h_cache) = h_cache_comps.get_mut(&entity) {
                if global_transform_changed {
                    let model_transform = transform_comps.get(&entity).unwrap();
                    h_cache.set_transform(&parent_h_cache, model_transform);
                    self.changed_h_caches.push(entity);
                }

                let relation = relation_comps.get(&entity).unwrap();
                h_cache.active = parent_h_cache.active && relation.active;

                *h_cache
            } else {
                parent_h_cache
            };

            if let Some(relation) = relation_comps.get(&entity) {
                // Because we're popping from the stack, insert in reverse order
                // to preserve the right order of insertion to `ordered_entities`
                stack.extend(relation.children.iter().rev().map(|e| StackEntry {
                    entity: *e,
                    parent_global_transform_changed: global_transform_changed,
                    parent_h_cache: global_h_cache,
                }));
            }
        }

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
