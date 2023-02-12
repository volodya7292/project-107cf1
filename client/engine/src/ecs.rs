use crate::ecs::dirty_components::DirtyComponents;
use crate::EngineContext;
use bumpalo::Bump;
use common::scene;
use common::scene::relation::Relation;
use entity_data::{Component, EntityId, EntityStorage, StaticArchetype, SystemAccess};
use std::any::TypeId;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;
use std::ptr;

pub mod component;
pub mod dirty_components;
pub(crate) mod system;

pub const N_MAX_OBJECTS: usize = 65535;

trait ApplyFn: FnOnce(DirectEntryAccess) {
    /// Safety: `self` must not be used in any way after calling this function.
    unsafe fn call(&mut self, access: DirectEntryAccess);
}

impl<F: FnOnce(DirectEntryAccess)> ApplyFn for F {
    unsafe fn call(&mut self, access: DirectEntryAccess) {
        ptr::read(self)(access)
    }
}

type ApplyDynFn = dyn ApplyFn<Output = ()>;

struct Modification {
    entity: EntityId,
    raw_apply_fn_ptr: *mut ApplyDynFn,
}

struct Modifications {
    heap: Bump,
    infos: Vec<Modification>,
}

pub trait SceneObject: StaticArchetype {}

/// Provides access to the `EntityStorage` with component change tracking.
pub struct SceneAccess<'a> {
    pub context: &'a EngineContext,
    pub storage: RefMut<'a, EntityStorage>,
    modifications: RefCell<Modifications>,
}

impl<'a> SceneAccess<'a> {
    pub fn new(context: &'a EngineContext) -> Self {
        let storage = context.storage.borrow_mut();
        Self {
            modifications: RefCell::new(Modifications {
                heap: Default::default(),
                infos: Vec::with_capacity(256),
            }),
            context,
            storage,
        }
    }

    pub fn add_object<O: SceneObject>(&mut self, parent: Option<EntityId>, object: O) -> Option<EntityId> {
        let mut obj_count = self.context.object_count.borrow_mut();
        let mut dirty_comps = self.context.dirty_comps.borrow_mut();

        if *obj_count >= N_MAX_OBJECTS {
            assert_eq!(*obj_count, N_MAX_OBJECTS);
            return None;
        }

        let comp_ids = object.component_ids();
        let entity = self.storage.add(object);

        for id in comp_ids {
            dirty_comps.add_with_component_id(id, &entity)
        }

        if let Some(parent) = parent {
            Self::add_children(&self.storage.access(), parent, &[entity]);
        }

        *obj_count += 1;

        let module_manager = self.context.module_manager.borrow();
        module_manager.on_object_added(&entity);

        Some(entity)
    }

    // TODO CORE: move to base
    /// Removes object and its children
    pub fn remove_object(&mut self, id: &EntityId) {
        let entities_to_remove = scene::collect_relation_tree(&self.storage.access(), id);
        let module_manager = self.context.module_manager.borrow();

        for entity in entities_to_remove {
            // Remove the entity from its parent's child list
            if let Some(relation) = self.storage.get::<Relation>(&entity) {
                let parent = relation.parent;

                if let Some(parent) = self.storage.get_mut::<Relation>(&parent) {
                    parent.children.remove(&entity);
                }
            }

            module_manager.on_object_remove(id);
            self.storage.remove(&entity);

            let mut obj_count = self.context.object_count.borrow_mut();
            *obj_count -= 1;
        }
    }

    // TODO CORE: move to base
    pub fn add_children(access: &SystemAccess, parent: EntityId, children: &[EntityId]) {
        let mut relation_comps = access.component_mut::<Relation>();

        for child in children {
            let relation = relation_comps
                .get_mut(child)
                .expect("child must have a Relation component");

            if relation.parent != EntityId::NULL {
                panic!("child already has a parent assigned");
            }

            relation.parent = parent;
        }

        let parent_relation = relation_comps
            .get_mut(&parent)
            .expect("parent must have Relation component");
        parent_relation.children.extend(children);
    }

    pub fn object_raw(&self, entity: &EntityId) -> Option<EntityAccess<()>> {
        Some(EntityAccess {
            context: &self.context,
            entry: self.storage.entry(entity)?,
            modifications: &self.modifications,
            _arch: Default::default(),
        })
    }

    pub fn object<A: StaticArchetype>(&self, entity: &EntityId) -> Option<EntityAccess<A>> {
        assert_eq!(
            self.storage.type_id_to_archetype_id(&TypeId::of::<A>()),
            Some(entity.archetype_id)
        );

        Some(EntityAccess {
            context: &self.context,
            entry: self.storage.entry(entity)?,
            modifications: &self.modifications,
            _arch: Default::default(),
        })
    }

    pub fn entry_mut(&mut self, entity: &EntityId) -> Option<DirectEntryAccess> {
        Some(DirectEntryAccess {
            dirty_components: self.context.dirty_comps.borrow_mut(),
            entry: self.storage.entry_mut(entity)?,
        })
    }

    /// Applies recorded entity modifications.
    pub fn apply_modifications(&mut self) {
        let mut mods = self.modifications.borrow_mut();

        for info in mods.infos.drain(..) {
            let entry = self.storage.entry_mut(&info.entity).unwrap();
            let f = unsafe { &mut *info.raw_apply_fn_ptr };

            let direct_access = DirectEntryAccess {
                dirty_components: self.context.dirty_comps.borrow_mut(),
                entry,
            };

            unsafe { f.call(direct_access) };
        }

        mods.heap.reset();
    }
}

impl Drop for SceneAccess<'_> {
    fn drop(&mut self) {
        self.apply_modifications();
    }
}

pub struct EntityAccess<'a, A> {
    context: &'a EngineContext,
    entry: entity_data::Entry<'a>,
    modifications: &'a RefCell<Modifications>,
    _arch: PhantomData<A>,
}

impl<'a, A> EntityAccess<'a, A> {
    pub fn context(&self) -> &'a EngineContext {
        self.context
    }

    pub fn get<C: Component>(&'a self) -> &'a C {
        self.entry.get().unwrap()
    }

    /// Records entity modifications, applies them after underlying `SceneAccess` is dropped.
    pub fn modify<F: FnOnce(DirectEntryAccess) + 'static>(&self, apply_fn: F) {
        let mut mods = self.modifications.borrow_mut();
        let raw_apply_fn_ptr = mods.heap.alloc(apply_fn) as *mut ApplyDynFn;

        mods.infos.push(Modification {
            entity: *self.entry.entity(),
            raw_apply_fn_ptr,
        });
    }
}

pub struct DirectEntryAccess<'a> {
    dirty_components: RefMut<'a, DirtyComponents>,
    entry: entity_data::EntryMut<'a>,
}

impl DirectEntryAccess<'_> {
    pub fn get<C: Component>(&self) -> Option<&C> {
        self.entry.get()
    }

    pub fn get_mut<C: Component>(&mut self) -> Option<&mut C> {
        self.dirty_components.add::<C>(self.entry.entity());
        self.entry.get_mut()
    }
}
