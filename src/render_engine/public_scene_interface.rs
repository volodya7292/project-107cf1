use crate::component;
use crate::component::Transform;
use crate::render_engine::scene::{
    ComponentStorage, ComponentStorageImpl, ComponentStorageMut, Entity, LockedStorage, Resources,
};
use crate::render_engine::Scene;
use parking_lot::RwLock;
use std::sync::Arc;

pub struct PublicSceneInterface {
    pub(super) scene: Scene,
    /// Global parent that encompasses all objects in the scene.
    /// It is needed to control global transformation of all the objects.
    /// To allow accurate rendering of distant objects beyond 32-bit float precision,
    /// camera is moved to (0, 0, 0) and all the objects are moved relatively to camera.
    /// So, camera position is always at the origin, and all the objects are placed relatively to camera.
    pub(super) global_parent: Entity,
}

impl PublicSceneInterface {
    pub(super) fn new() -> Self {
        let mut scene = Scene::new();

        scene.prepare_storage::<component::Parent>();
        scene.prepare_storage::<component::Children>();
        scene.prepare_storage::<component::Transform>().emit_events(true);
        scene
            .prepare_storage::<component::WorldTransform>()
            .emit_events(true);
        scene.prepare_storage::<component::Renderer>().emit_events(true);
        scene.prepare_storage::<component::VertexMesh>().emit_events(true);
        scene.prepare_storage::<component::Camera>();

        let global_parent = scene.create_entity();
        scene
            .storage_write::<Transform>()
            .set(global_parent, Transform::default());

        Self { scene, global_parent }
    }

    pub fn add_resource<T: 'static + Send + Sync>(&self, resource: T) {
        self.scene.add_resource(resource);
    }

    /// When creating multiple entities, prefer doing it through `self.entities()`.
    pub fn create_entity(&self) -> Entity {
        let entity = self.scene.create_entity();
        self.relations_mut().add_children(self.global_parent, &[entity]);
        entity
    }

    pub fn create_entities(&self, n: u32) -> Vec<Entity> {
        let mut entity_storage = self.scene.entities().write();
        let entities: Vec<_> = (0..n).map(|_| entity_storage.create()).collect();

        drop(entity_storage);
        self.relations_mut().add_children(self.global_parent, &entities);

        entities
    }

    /// Remove entities and their children recursively.
    pub fn remove_entities(&self, entities: &[Entity]) {
        let child_comps = self.storage_read::<component::Children>();
        let mut total_entities = Vec::with_capacity(child_comps.len());

        total_entities.extend(entities);

        let relations = self.relations_mut();

        for &entity in entities {
            relations.collect_children_recursively(&mut total_entities, entity);
        }

        // Note: entities are not removed from their parents Children components,
        // they are just becoming dead (not alive), and removed when new children are set

        drop(child_comps);
        self.scene.remove_entities(&total_entities);
    }

    pub fn resources(&self) -> Arc<RwLock<Resources>> {
        self.scene.resources()
    }

    pub fn prepare_storage<T>(&mut self) -> ComponentStorageMut<T>
    where
        T: 'static + Send + Sync,
    {
        self.scene.prepare_storage::<T>()
    }

    pub fn storage<T>(&self) -> LockedStorage<T>
    where
        T: 'static + Send + Sync,
    {
        self.scene.storage::<T>()
    }

    pub fn storage_read<T>(&self) -> ComponentStorage<T>
    where
        T: 'static + Send + Sync,
    {
        self.scene.storage_read::<T>()
    }

    pub fn storage_write<T>(&self) -> ComponentStorageMut<T>
    where
        T: 'static + Send + Sync,
    {
        self.scene.storage_write::<T>()
    }

    pub fn relations_mut(&self) -> RelationInterfaceMut {
        RelationInterfaceMut {
            parent_comps: self.scene.storage_write::<component::Parent>(),
            children_comps: self.scene.storage_write::<component::Children>(),
        }
    }
}

pub struct RelationInterfaceMut<'a> {
    parent_comps: ComponentStorageMut<'a, component::Parent>,
    children_comps: ComponentStorageMut<'a, component::Children>,
}

impl RelationInterfaceMut<'_> {
    pub(super) fn collect_children_recursively(&self, children: &mut Vec<Entity>, entity: Entity) {
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
            self.parent_comps.set(child, component::Parent(parent));
        }

        children_comp.children.extend(children);
    }

    /// Sets children and accordingly adds `Parent` component to each child
    pub fn set_children(&mut self, parent: Entity, children: &[Entity]) {
        self.children_comps.set(
            parent,
            component::Children {
                children: children.iter().cloned().collect(),
            },
        );

        for &child in children {
            if self.parent_comps.contains(child) {
                panic!("child already has a parent assigned");
            }
            self.parent_comps.set(child, component::Parent(parent));
        }
    }
}
