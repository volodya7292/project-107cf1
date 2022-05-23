use crate::ecs::component;
use crate::ecs::component::internal::{Children, GlobalTransform, Parent};
use crate::ecs::scene_storage;
use crate::ecs::scene_storage::{ComponentStorageImpl, Entity};

// Propagates transform hierarchy and calculates global transforms
pub(crate) struct HierarchyPropagation<'a> {
    pub parent_comps: scene_storage::LockedStorage<'a, Parent>,
    pub children_comps: scene_storage::LockedStorage<'a, Children>,
    pub transform_comps: scene_storage::LockedStorage<'a, component::Transform>,
    pub global_transform_comps: scene_storage::LockedStorage<'a, GlobalTransform>,
    pub ordered_entities: &'a mut Vec<Entity>,
}

struct StackEntry {
    entity: Entity,
    parent_global_transform_changed: bool,
    parent_global_transform: GlobalTransform,
}

impl HierarchyPropagation<'_> {
    pub fn run(&mut self) {
        let parent_comps = self.parent_comps.read();
        let children_comps = self.children_comps.read();
        let mut transform_comps = self.transform_comps.write();
        let mut global_transform_comps = self.global_transform_comps.write();
        let mut stack = Vec::<StackEntry>::with_capacity(transform_comps.len());

        // Collect global parents
        // !Parent & Transform (global parent entity doesn't have a Parent component)
        let entities = transform_comps.entries().difference(&parent_comps);
        stack.extend(entities.iter().map(|e| StackEntry {
            entity: e,
            parent_global_transform_changed: false,
            parent_global_transform: Default::default(),
        }));

        self.ordered_entities.clear();

        // Recursion using loop
        while let Some(StackEntry {
            entity,
            parent_global_transform_changed: parent_transform_changed,
            parent_global_transform,
        }) = stack.pop()
        {
            // Maybe this entity is dead (was removed but not removed from parent's `Children` component)
            if !transform_comps.contains(entity) {
                continue;
            }

            self.ordered_entities.push(entity);

            let global_transform_changed = parent_transform_changed || transform_comps.was_modified(entity);

            let global_transform = if global_transform_changed {
                let model_transform = transform_comps.get(entity).unwrap();

                let new_global_transform =
                    GlobalTransform::new(parent_global_transform.transform.combine(model_transform));

                global_transform_comps.set(entity, new_global_transform);
                new_global_transform
            } else {
                *global_transform_comps.get(entity).unwrap()
            };

            if let Some(children) = children_comps.get(entity) {
                // Because we're popping from the stack, insert in reversed order
                // to preserve the right order of insertion to `ordered_entities`
                stack.extend(children.children.iter().rev().map(|e| StackEntry {
                    entity: *e,
                    parent_global_transform_changed: global_transform_changed,
                    parent_global_transform: global_transform,
                }));
            }
        }

        transform_comps.clear_events();
    }
}
