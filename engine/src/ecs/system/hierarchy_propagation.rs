use crate::ecs::component;
use crate::ecs::component::internal::{Children, Parent, WorldTransform};
use crate::ecs::scene;
use crate::ecs::scene::{ComponentStorageImpl, Entity};

// Propagates transform hierarchy and calculates world transforms
pub(crate) struct HierarchyPropagation<'a> {
    pub parent_comps: scene::LockedStorage<'a, Parent>,
    pub children_comps: scene::LockedStorage<'a, Children>,
    pub transform_comps: scene::LockedStorage<'a, component::Transform>,
    pub world_transform_comps: scene::LockedStorage<'a, WorldTransform>,
    pub ordered_entities: &'a mut Vec<Entity>,
}

struct StackEntry {
    entity: Entity,
    parent_world_transform_changed: bool,
    parent_world_transform: WorldTransform,
}

impl HierarchyPropagation<'_> {
    pub fn run(&mut self) {
        let parent_comps = self.parent_comps.read();
        let children_comps = self.children_comps.read();
        let mut transform_comps = self.transform_comps.write();
        let mut world_transform_comps = self.world_transform_comps.write();
        let mut stack = Vec::<StackEntry>::with_capacity(transform_comps.len());

        // Collect global parents
        // !Parent & Transform (global parent entity doesn't have a Parent component)
        let entities = transform_comps.entries().difference(&parent_comps);
        stack.extend(entities.iter().map(|e| StackEntry {
            entity: e,
            parent_world_transform_changed: false,
            parent_world_transform: Default::default(),
        }));

        self.ordered_entities.clear();

        // Recursion using loop
        while let Some(StackEntry {
            entity,
            parent_world_transform_changed: parent_transform_changed,
            parent_world_transform,
        }) = stack.pop()
        {
            // Maybe this entity is dead (was removed but not removed from parent's `Children` component)
            if !transform_comps.contains(entity) {
                continue;
            }

            self.ordered_entities.push(entity);

            let world_transform_changed = parent_transform_changed || transform_comps.was_modified(entity);

            let world_transform = if world_transform_changed {
                let model_transform = transform_comps.get(entity).unwrap();

                let new_world_transform: WorldTransform =
                    parent_world_transform.combine(model_transform).into();

                world_transform_comps.set(entity, new_world_transform);
                new_world_transform
            } else {
                *world_transform_comps.get(entity).unwrap()
            };

            if let Some(children) = children_comps.get(entity) {
                // Because we're popping from the stack, insert in reversed order
                // to preserve the right order of insertion to `ordered_entities`
                stack.extend(children.children.iter().rev().map(|e| StackEntry {
                    entity: *e,
                    parent_world_transform_changed: world_transform_changed,
                    parent_world_transform: world_transform,
                }));
            }
        }

        transform_comps.clear_events();
    }
}
