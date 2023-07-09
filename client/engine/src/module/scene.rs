pub mod change_manager;

use crate::ecs::component::SceneEventHandler;
use crate::module::scene::change_manager::{ChangeType, ComponentChangesHandle};
use crate::module::EngineModule;
use crate::EngineContext;
use change_manager::SceneChangeManager;
use common::any::AsAny;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRef, OwnedRefMut};
use common::scene::relation::Relation;
use common::types::{HashMap, HashSet};
use entity_data::{Component, EntityId, EntityStorage, StaticArchetype};
use std::any::TypeId;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;
use std::ops::Deref;

pub const N_MAX_OBJECTS: usize = 65535;

pub trait SceneObject: StaticArchetype {
    fn request_update_on_addition() -> bool {
        false
    }
}

pub type EntityUpdateJob = fn(entity: &EntityId, scene: &mut Scene, &EngineContext);

pub trait Resource: AsAny {}

impl<T: 'static> Resource for T {}

pub struct Scene {
    storage: EntityStorage,
    object_count: usize,
    change_manager: Lrc<SceneChangeManager>,
    entities_to_update: HashSet<EntityId>,
    component_changes: HashMap<TypeId, ComponentChangesHandle>,
    resources: RefCell<HashMap<TypeId, Lrc<dyn Resource>>>,
}

impl Scene {
    pub fn new() -> Self {
        let change_manager = SceneChangeManager::new();
        Self {
            storage: Default::default(),
            object_count: 0,
            change_manager: Lrc::wrap(change_manager),
            entities_to_update: HashSet::with_capacity(1024),
            component_changes: Default::default(),
            resources: RefCell::new(Default::default()),
        }
    }

    pub fn add_resource<R: Resource>(&self, resource: R) {
        self.resources
            .borrow_mut()
            .insert(TypeId::of::<R>(), Lrc::wrap(resource));
    }

    pub fn resource<R: Resource>(&self) -> OwnedRef<dyn Resource, R> {
        let res = self.resources.borrow().get(&TypeId::of::<R>()).unwrap().clone();
        OwnedRef::map(res.borrow_owned(), |v| v.as_any().downcast_ref::<R>().unwrap())
    }

    pub fn storage(&self) -> &EntityStorage {
        &self.storage
    }

    pub fn storage_mut(&mut self) -> &mut EntityStorage {
        &mut self.storage
    }

    pub fn entry_checked(&mut self, entity: &EntityId) -> Option<EntityAccess<()>> {
        Some(EntityAccess {
            entry: self.storage.entry_mut(entity)?,
            change_manager: self.change_manager.borrow_mut(),
            entities_to_update: &mut self.entities_to_update,
            _arch: Default::default(),
        })
    }

    pub fn entry(&mut self, entity: &EntityId) -> EntityAccess<()> {
        self.entry_checked(entity).unwrap()
    }

    pub fn object<T: StaticArchetype>(&mut self, entity: &ObjectEntityId<T>) -> EntityAccess<T> {
        assert_eq!(
            self.storage.type_id_to_archetype_id(&TypeId::of::<T>()),
            Some(entity.archetype_id)
        );

        EntityAccess {
            entry: self.storage.entry_mut(entity).unwrap(),
            change_manager: self.change_manager.borrow_mut(),
            entities_to_update: &mut self.entities_to_update,
            _arch: Default::default(),
        }
    }

    pub fn add_object<O: SceneObject>(
        &mut self,
        parent: Option<EntityId>,
        object: O,
    ) -> Option<ObjectEntityId<O>> {
        let obj_count = &mut self.object_count;
        let mut change_manager = self.change_manager.borrow_mut();

        if *obj_count >= N_MAX_OBJECTS {
            assert_eq!(*obj_count, N_MAX_OBJECTS);
            return None;
        }

        let comp_ids = object.component_ids();
        let entity = self.storage.add(object);

        *obj_count += 1;

        // Add component changes
        for comp_id in comp_ids {
            change_manager.record(comp_id, entity, ChangeType::Added);
        }

        // Add new component change flows if necessary
        if let Some(event_handler) = self.storage.get::<SceneEventHandler>(&entity) {
            for comp_id in event_handler.on_component_update.keys() {
                self.component_changes
                    .entry(*comp_id)
                    .or_insert_with(|| change_manager.register_component_flow_for_id(*comp_id));
            }
        }

        drop(change_manager);

        if let Some(parent) = parent {
            self.add_children(parent, &[entity]);
        }

        if O::request_update_on_addition() {
            self.entities_to_update.insert(entity);
        }

        Some(ObjectEntityId::new(entity))
    }

    // TODO CORE: move to base
    /// Removes object and its children
    pub fn remove_object(&mut self, id: &EntityId) {
        let entities_to_remove = common::scene::collect_relation_tree(&self.storage.access(), id);
        let mut change_manager = self.change_manager.borrow_mut();

        for entity in entities_to_remove {
            // Remove the entity from its parent's child list
            if let Some(relation) = self.storage.get::<Relation>(&entity) {
                let parent = relation.parent;

                if let Some(parent) = self.storage.get_mut::<Relation>(&parent) {
                    parent.children.remove(&entity);
                }
            }

            // Remove component changes if any present
            let arch = self.storage.get_archetype_by_id(entity.archetype_id).unwrap();
            for info in arch.iter_component_infos() {
                change_manager.record(info.type_id, entity, ChangeType::Removed);
            }

            self.storage.remove(&entity);
            self.object_count -= 1;

            self.entities_to_update.remove(&entity);
        }
    }

    pub fn add_children(&mut self, parent: EntityId, children: &[EntityId]) {
        for child in children {
            let relation = self
                .storage
                .get_mut::<Relation>(child)
                .expect("child must have a Relation component");

            if relation.parent != EntityId::NULL {
                panic!("child already has a parent assigned");
            }
            if *child == parent {
                panic!("child must not be parent itself");
            }

            relation.parent = parent;
        }

        let parent_relation = self
            .storage
            .get_mut::<Relation>(&parent)
            .expect("parent must have Relation component");

        parent_relation.children.extend(children);
    }

    pub fn change_manager(&self) -> &Lrc<SceneChangeManager> {
        &self.change_manager
    }

    pub fn change_manager_mut(&self) -> OwnedRefMut<SceneChangeManager, SceneChangeManager> {
        self.change_manager.borrow_mut_owned()
    }
}

impl EngineModule for Scene {
    fn on_update(&mut self, dt: f64, ctx: &EngineContext) {
        let to_update: Vec<_> = self.entities_to_update.drain().collect();
        let comp_changes: HashMap<_, _> = self
            .component_changes
            .iter()
            .map(|(comp_id, handle)| {
                (
                    *comp_id,
                    self.change_manager.borrow_mut().take_new::<Vec<_>>(*handle),
                )
            })
            .collect();

        for (comp_id, changes) in comp_changes {
            for entity in changes {
                let Some(event_handler) = self.storage.get::<SceneEventHandler>(&entity) else {
                    continue;
                };
                if let Some(on_update) = event_handler.on_component_update(&comp_id) {
                    on_update(&entity, self, ctx);
                }
            }
        }

        for entity in to_update {
            let event_handler = self.storage.get::<SceneEventHandler>(&entity).unwrap();
            let on_update = event_handler.on_update();
            on_update(&entity, self, ctx, dt);
        }
    }
}

pub struct EntityAccess<'a, A> {
    entry: entity_data::EntryMut<'a>,
    change_manager: RefMut<'a, SceneChangeManager>,
    entities_to_update: &'a mut HashSet<EntityId>,
    _arch: PhantomData<A>,
}

impl<A> EntityAccess<'_, A> {
    pub fn entity(&self) -> &EntityId {
        self.entry.entity()
    }

    #[inline]
    pub fn get_checked<C: Component>(&self) -> Option<&C> {
        self.entry.get()
    }

    #[inline]
    pub fn get<C: Component>(&self) -> &C {
        self.get_checked().unwrap()
    }

    #[inline]
    pub fn get_mut_checked<C: Component>(&mut self) -> Option<&mut C> {
        self.change_manager.record_modification::<C>(*self.entry.entity());
        self.entry.get_mut::<C>()
    }

    #[inline]
    pub fn get_mut<C: Component>(&mut self) -> &mut C {
        self.get_mut_checked().unwrap()
    }

    pub fn request_update(&mut self) {
        self.entities_to_update.insert(*self.entry.entity());
    }
}

pub struct ObjectEntityId<T: ?Sized> {
    entity_id: EntityId,
    _ty: PhantomData<T>,
}

impl<T: StaticArchetype> ObjectEntityId<T> {
    pub fn new(entity_id: EntityId) -> Self {
        Self {
            entity_id,
            _ty: Default::default(),
        }
    }
}

impl<T: StaticArchetype> Deref for ObjectEntityId<T> {
    type Target = EntityId;

    fn deref(&self) -> &Self::Target {
        &self.entity_id
    }
}

impl<T> From<EntityId> for ObjectEntityId<T> {
    fn from(value: EntityId) -> Self {
        Self {
            entity_id: value,
            _ty: Default::default(),
        }
    }
}

impl<T> From<&EntityId> for ObjectEntityId<T> {
    fn from(value: &EntityId) -> Self {
        Self {
            entity_id: *value,
            _ty: Default::default(),
        }
    }
}

impl<T> Clone for ObjectEntityId<T> {
    fn clone(&self) -> Self {
        Self {
            entity_id: self.entity_id,
            _ty: Default::default(),
        }
    }
}

impl<T> Copy for ObjectEntityId<T> {}

impl<T> Default for ObjectEntityId<T> {
    fn default() -> Self {
        Self {
            entity_id: Default::default(),
            _ty: Default::default(),
        }
    }
}
