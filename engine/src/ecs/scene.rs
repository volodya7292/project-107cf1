use crate::ecs::component;
use crate::ecs::component::internal::{Children, Parent, WorldTransform};
use crate::ecs::component::Transform;
use crate::ecs::scene_storage::SceneStorage;
use crate::ecs::scene_storage::{
    ComponentStorage, ComponentStorageImpl, ComponentStorageMut, LockedStorage, Resources,
};
use parking_lot::RwLock;
use std::sync::Arc;

pub use crate::ecs::scene_storage::Entity;

pub struct Scene {
    inner: SceneStorage,
    /// Global parent that encompasses all objects in the scene.
    /// It is needed to control global transformation of all the objects.
    /// To allow accurate rendering of distant objects beyond 32-bit float precision,
    /// camera is moved to (0, 0, 0) and all the objects are moved relatively to camera.
    /// So, camera position is always at the origin, and all the objects are placed relatively to camera.
    global_parent: Entity,
}

impl Scene {
    pub(crate) fn new() -> Self {
        let mut scene = SceneStorage::new();

        scene.prepare_storage::<Parent>();
        scene.prepare_storage::<Children>();
        scene.prepare_storage::<component::Transform>().emit_events(true);
        scene.prepare_storage::<WorldTransform>().emit_events(true);
        scene
            .prepare_storage::<component::RenderConfig>()
            .emit_events(true);
        scene.prepare_storage::<component::VertexMesh>().emit_events(true);

        let global_parent = scene.create_entity();
        scene
            .storage_write::<Transform>()
            .set(global_parent, Transform::default());

        Self {
            inner: scene,
            global_parent,
        }
    }

    pub(crate) fn global_transform(&self) -> Transform {
        *self.storage_read::<Transform>().get(self.global_parent).unwrap()
    }

    pub(crate) fn set_global_transform(&self, transform: Transform) {
        *self
            .storage_write::<Transform>()
            .get_mut(self.global_parent)
            .unwrap() = transform;
    }

    pub fn add_resource<T: 'static + Send + Sync>(&self, resource: T) {
        self.inner.add_resource(resource);
    }

    /// When creating multiple entities, prefer doing it through `self.entities()`.
    pub fn create_entity(&self) -> Entity {
        let entity = self.inner.create_entity();
        self.relations_mut().add_children(self.global_parent, &[entity]);
        entity
    }

    pub fn create_entities(&self, n: u32) -> Vec<Entity> {
        let mut entity_storage = self.inner.entities().write();
        let entities: Vec<_> = (0..n).map(|_| entity_storage.create()).collect();

        drop(entity_storage);
        self.relations_mut().add_children(self.global_parent, &entities);

        entities
    }

    /// Remove entities and their children recursively.
    pub fn remove_entities(&self, entities: &[Entity]) {
        let relations = self.relations_mut();
        let mut total_entities = Vec::with_capacity(relations.children_comps.len());

        total_entities.extend(entities);

        for &entity in entities {
            relations.collect_children_recursively(&mut total_entities, entity);
        }

        // Note: entities are not removed from their parents Children components,
        // they are just becoming dead (not alive), and removed when new children are set

        drop(relations);
        self.inner.remove_entities(&total_entities);
    }

    pub fn resources(&self) -> Arc<RwLock<Resources>> {
        self.inner.resources()
    }

    pub fn prepare_storage<T>(&mut self) -> ComponentStorageMut<T>
    where
        T: 'static + Send + Sync,
    {
        self.inner.prepare_storage::<T>()
    }

    pub fn storage<T>(&self) -> LockedStorage<T>
    where
        T: 'static + Send + Sync,
    {
        self.inner.storage::<T>()
    }

    pub fn storage_read<T>(&self) -> ComponentStorage<T>
    where
        T: 'static + Send + Sync,
    {
        self.inner.storage_read::<T>()
    }

    pub fn storage_write<T>(&self) -> ComponentStorageMut<T>
    where
        T: 'static + Send + Sync,
    {
        self.inner.storage_write::<T>()
    }

    pub fn relations_mut(&self) -> RelationInterfaceMut {
        RelationInterfaceMut {
            parent_comps: self.inner.storage_write::<Parent>(),
            children_comps: self.inner.storage_write::<Children>(),
        }
    }
}

pub struct RelationInterfaceMut<'a> {
    parent_comps: ComponentStorageMut<'a, Parent>,
    children_comps: ComponentStorageMut<'a, Children>,
}

impl RelationInterfaceMut<'_> {
    pub(crate) fn collect_children_recursively(&self, children: &mut Vec<Entity>, entity: Entity) {
        let mut stack = Vec::<Entity>::with_capacity(self.children_comps.len());
        stack.push(entity);

        while let Some(entity) = stack.pop() {
            if let Some(children_comp) = self.children_comps.get(entity) {
                children.extend(&children_comp.children);
                stack.extend(&children_comp.children);
            }
        }
    }

    pub fn add_children(&mut self, parent: Entity, children: &[Entity]) {
        let children_comp = self.children_comps.get_or_insert_default(parent);

        for &child in children {
            if self.parent_comps.contains(child) {
                panic!("child already has a parent assigned");
            }
            self.parent_comps.set(child, Parent(parent));
        }

        children_comp.children.extend(children);
    }

    /// Sets children and accordingly adds `Parent` component to each child
    pub fn set_children(&mut self, parent: Entity, children: &[Entity]) {
        self.children_comps.set(
            parent,
            Children {
                children: children.iter().cloned().collect(),
            },
        );

        for &child in children {
            if self.parent_comps.contains(child) {
                panic!("child already has a parent assigned");
            }
            self.parent_comps.set(child, Parent(parent));
        }
    }
}
