pub mod change_manager;

use crate::ecs::component::SceneEventHandler;
use crate::module::scene::change_manager::{ChangeType, ComponentChangesHandle};
use crate::module::EngineModule;
use crate::EngineContext;
use change_manager::SceneChangeManager;
use common::lrc::{Lrc, LrcExt, LrcExtSized, OwnedRefMut};
use common::scene::relation::Relation;
use common::types::HashSet;
use entity_data::{Component, EntityId, EntityStorage, StaticArchetype};
use std::any::TypeId;
use std::cell::RefMut;
use std::marker::PhantomData;

pub const N_MAX_OBJECTS: usize = 65535;

pub trait SceneObject: StaticArchetype {}

pub struct Scene {
    storage: EntityStorage,
    object_count: usize,
    change_manager: Lrc<SceneChangeManager>,
    scene_event_handler_changes: ComponentChangesHandle,
    on_update_entities: HashSet<EntityId>,
}

impl Scene {
    pub fn new() -> Self {
        let mut change_manager = SceneChangeManager::new();
        let scene_event_handler_changes = change_manager.register_component_flow::<SceneEventHandler>();

        Self {
            storage: Default::default(),
            object_count: 0,
            change_manager: Lrc::wrap(change_manager),
            scene_event_handler_changes,
            on_update_entities: HashSet::with_capacity(1024),
        }
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
            _arch: Default::default(),
        })
    }

    pub fn entry(&mut self, entity: &EntityId) -> EntityAccess<()> {
        self.entry_checked(entity).unwrap()
    }

    pub fn object<A: StaticArchetype>(&mut self, entity: &EntityId) -> Option<EntityAccess<A>> {
        assert_eq!(
            self.storage.type_id_to_archetype_id(&TypeId::of::<A>()),
            Some(entity.archetype_id)
        );

        Some(EntityAccess {
            entry: self.storage.entry_mut(entity)?,
            change_manager: self.change_manager.borrow_mut(),
            _arch: Default::default(),
        })
    }

    // TODO: add entity `generation` property
    pub fn add_object<O: SceneObject>(&mut self, parent: Option<EntityId>, object: O) -> Option<EntityId> {
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
        drop(change_manager);

        if let Some(parent) = parent {
            self.add_children(parent, &[entity]);
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
                    parent.children.remove(&entity);
                }
            }

            // Remove component changes if any present
            let arch = self.storage.get_archetype_by_id(entity.archetype_id).unwrap();
            for ty in arch.iter_component_types() {
                change_manager.record(*ty, entity, ChangeType::Removed);
            }

            self.storage.remove(&entity);
            self.object_count -= 1;
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
    fn on_update(&mut self, ctx: &EngineContext) {
        let changes: Vec<_> = self.change_manager_mut().take(self.scene_event_handler_changes);

        for change in changes {
            let entity = change.entity();

            if change.ty() == ChangeType::Removed {
                self.on_update_entities.remove(entity);
                continue;
            }

            let entry = self.entry_checked(&entity).unwrap();
            let event_handler = *entry.get::<SceneEventHandler>();
            let on_update = event_handler.on_update();

            drop(entry);

            if event_handler.on_update_active {
                on_update(entity, self, ctx);
            } else {
                self.on_update_entities.remove(entity);
            }
        }
    }
}

pub struct EntityAccess<'a, A> {
    entry: entity_data::EntryMut<'a>,
    change_manager: RefMut<'a, SceneChangeManager>,
    _arch: PhantomData<A>,
}

impl<A> EntityAccess<'_, A> {
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
}
