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

#[derive(Copy, Clone)]
struct ParentInfo {
    global_transform_changed: bool,
    h_cache: HierarchyCacheC,
}

impl SystemHandler for HierarchyPropagation<'_> {
    fn run(&mut self, data: SystemAccess) {
        let t0 = Instant::now();
        let relation_comps = data.component::<Relation>();
        let transform_comps = data.component::<TransformC>();
        let mut h_cache_comps = data.component_mut::<HierarchyCacheC>();

        self.ordered_entities.clear();

        scene::walk_relation_tree(
            &data,
            &self.root_entity,
            ParentInfo {
                global_transform_changed: false,
                h_cache: Default::default(),
            },
            |entity, parent_info| {
                self.ordered_entities.push(*entity);

                let global_transform_changed =
                    parent_info.global_transform_changed || self.dirty_transforms.contains(entity);

                let global_h_cache = if let Some(h_cache) = h_cache_comps.get_mut(entity) {
                    if global_transform_changed {
                        let model_transform = transform_comps.get(entity).unwrap();
                        h_cache.set_transform(&parent_info.h_cache, model_transform);
                        self.changed_h_caches.push(*entity);
                    }

                    let relation = relation_comps.get(entity).unwrap();
                    h_cache.active = parent_info.h_cache.active && relation.active;

                    *h_cache
                } else {
                    parent_info.h_cache
                };

                ParentInfo {
                    global_transform_changed,
                    h_cache: global_h_cache,
                }
            },
        );

        let t1 = Instant::now();
        self.run_time = (t1 - t0).as_secs_f64();
    }
}
