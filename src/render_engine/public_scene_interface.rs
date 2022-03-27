use crate::component;
use crate::component::WorldTransform;
use crate::render_engine::scene::{
    ComponentStorage, ComponentStorageImpl, ComponentStorageMut, Entities, Entity, LockedStorage, Resources,
};
use crate::render_engine::Scene;
use crate::utils;
use parking_lot::RwLock;
use std::any::TypeId;
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
        let scene = Scene::new();
        let global_parent = scene.create_entity();
        Self { scene, global_parent }
    }

    pub fn add_resource<T: 'static + Send + Sync>(&self, resource: T) {
        self.scene.add_resource(resource);
    }

    /// When creating multiple entities, prefer doing it through `self.entities()`.
    pub fn create_entity(&self) -> Entity {
        self.scene.create_entity()
    }

    /// Remove entities and their children recursively.
    pub fn remove_entities(&self, entities: &[Entity]) {
        // let parent_comps = self.storage_read::<component::Parent>();
        let child_comps = self.storage_read::<component::Children>();
        let mut total_entities = Vec::with_capacity(child_comps.len());

        total_entities.extend(entities);

        for &entity in entities {
            component::collect_children_recursively(&mut total_entities, entity, &child_comps);
        }

        // Note: entities are not removed from their Parents' Children components,
        // they are just becoming dead (not alive), and removed when new children are set

        drop(child_comps);
        self.scene.remove_entities(&total_entities);
    }

    pub fn resources(&self) -> Arc<RwLock<Resources>> {
        self.scene.resources()
    }

    pub fn entities(&self) -> &Arc<RwLock<Entities>> {
        self.scene.entities()
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
}
