pub mod change_manager;

use crate::ecs::component::scene_event_handler::OnUpdateCallback;
use crate::ecs::component::SceneEventHandler;
use crate::module::scene::change_manager::{ChangeType, ComponentChangesHandle};
use crate::module::EngineModule;
use crate::EngineContext;
use change_manager::SceneChangeManager;
use common::any::AsAny;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRef, OwnedRefMut};
use common::scene::relation::Relation;
use common::types::HashMap;
use entity_data::{Component, EntityId, EntityStorage, StaticArchetype};
use smallvec::SmallVec;
use std::any::TypeId;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;
use std::ops::Deref;

pub const N_MAX_OBJECTS: usize = 65535;

pub trait SceneObject: StaticArchetype {
    fn request_update_on_addition() -> bool {
        false
    }

    fn on_update(_entity: &EntityId, _ctx: &EngineContext, _dt: f64) {}
}

impl SceneObject for () {}

pub trait Resource: AsAny {}

impl<T: 'static> Resource for T {}

pub struct Scene {
    storage: EntityStorage,
    object_count: usize,
    change_manager: Lrc<SceneChangeManager>,
    entity_updaters: HashMap<EntityId, SmallVec<[OnUpdateCallback; 4]>>,
    component_changes: HashMap<TypeId, ComponentChangesHandle>,
    resources: RefCell<HashMap<TypeId, Lrc<dyn Resource>>>,
    named_resources: RefCell<HashMap<String, Lrc<dyn Resource>>>,
}

impl Scene {
    pub fn new() -> Self {
        let change_manager = SceneChangeManager::new();
        Self {
            storage: Default::default(),
            object_count: 0,
            change_manager: Lrc::wrap(change_manager),
            entity_updaters: HashMap::with_capacity(1024),
            component_changes: Default::default(),
            resources: RefCell::new(Default::default()),
            named_resources: RefCell::new(Default::default()),
        }
    }

    pub fn register_resource<R: Resource>(&self, resource: R) {
        self.resources
            .borrow_mut()
            .insert(TypeId::of::<R>(), Lrc::wrap(resource));
    }

    pub fn register_named_resource<R: Resource>(&self, name: impl Into<String>, resource: R) {
        self.named_resources
            .borrow_mut()
            .insert(name.into(), Lrc::wrap(resource));
    }

    pub fn resource<R: Resource>(&self) -> OwnedRef<dyn Resource, R> {
        let res = self.resources.borrow().get(&TypeId::of::<R>()).unwrap().clone();
        OwnedRef::map(res.borrow_owned(), |v| v.as_any().downcast_ref::<R>().unwrap())
    }

    pub fn named_resource<R: Resource>(&self, name: &str) -> OwnedRef<dyn Resource, R> {
        let res = self.named_resources.borrow().get(name).unwrap().clone();
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
            entity_updaters: &mut self.entity_updaters,
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
            entity_updaters: &mut self.entity_updaters,
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

        let entity = ObjectEntityId::new(entity);
        if O::request_update_on_addition() {
            self.object(&entity).request_update();
        }

        Some(entity)
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
                    parent.remove_child(&entity);
                }
            }

            // Remove component changes if any present
            let arch = self.storage.get_archetype_by_id(entity.archetype_id).unwrap();
            for info in arch.iter_component_infos() {
                change_manager.record(info.type_id, entity, ChangeType::Removed);
            }

            self.storage.remove(&entity);
            self.object_count -= 1;

            self.entity_updaters.remove(&entity);
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

        parent_relation.add_children(children.iter().cloned());
    }

    pub fn clear_children(&mut self, parent: &EntityId) {
        let parent_relation = self
            .storage
            .get_mut::<Relation>(parent)
            .expect("parent must have Relation component");

        let children: Vec<_> = parent_relation.unordered_children().collect();
        parent_relation.clear_children();

        for child in &children {
            self.remove_object(child);
        }
    }

    pub fn change_manager(&self) -> &Lrc<SceneChangeManager> {
        &self.change_manager
    }

    pub fn change_manager_mut(&self) -> OwnedRefMut<SceneChangeManager, SceneChangeManager> {
        self.change_manager.borrow_mut_owned()
    }
}

impl EngineModule for Scene {
    fn on_update(&mut self, _: f64, ctx: &EngineContext) {
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

        let storage = &self.storage;
        let component_callbacks: Vec<_> = comp_changes
            .iter()
            .flat_map(|(comp_id, entities)| {
                entities.iter().filter_map(move |entity| {
                    let Some(event_handler) = storage.get::<SceneEventHandler>(entity) else {
                        return None;
                    };
                    let Some(on_comp_update) = event_handler.on_component_update(&comp_id) else {
                        return None;
                    };
                    Some((*entity, *on_comp_update))
                })
            })
            .collect();
        let entity_updaters: Vec<_> = self.entity_updaters.drain().collect();

        ctx.dispatch_callback(move |ctx, _| {
            for (entity, callback) in component_callbacks {
                callback(&entity, ctx);
            }
        });

        ctx.dispatch_callback(move |ctx, dt| {
            for (entity, on_update_callbacks) in entity_updaters {
                for on_update in on_update_callbacks {
                    on_update(&entity, ctx, dt)
                }
            }
        });
    }
}

pub struct EntityAccess<'a, A> {
    entry: entity_data::EntryMut<'a>,
    change_manager: RefMut<'a, SceneChangeManager>,
    entity_updaters: &'a mut HashMap<EntityId, SmallVec<[OnUpdateCallback; 4]>>,
    _arch: PhantomData<A>,
}

impl<T: SceneObject> EntityAccess<'_, T> {
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

    pub fn request_custom_update(&mut self, f: OnUpdateCallback) {
        let entity_id = *self.entity();
        let updaters = self.entity_updaters.entry(entity_id).or_default();
        if !updaters.contains(&f) {
            updaters.push(f);
        }
    }

    pub fn request_update(&mut self) {
        if TypeId::of::<T>() == TypeId::of::<()>() {
            panic!("Invalid object type ()");
        }
        self.request_custom_update(T::on_update);
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
