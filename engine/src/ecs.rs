use crate::renderer::DirtyComponents;
use bumpalo::Bump;
use entity_data::{Component, EntityId, EntityStorage, StaticArchetype};
use std::any::TypeId;
use std::cell::{RefCell, RefMut};
use std::marker::PhantomData;
use std::ptr;

pub mod component;
pub mod dirty_components;
pub(crate) mod system;

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

/// Provides access to the `EntityStorage` with component change tracking.
pub struct SceneAccess<'a, Context> {
    pub storage: &'a mut EntityStorage,
    pub dirty_components: &'a RefCell<DirtyComponents>,
    modifications: RefCell<Modifications>,
    context: RefCell<Context>,
}

impl<'a, Ctx> SceneAccess<'a, Ctx> {
    pub fn new(
        storage: &'a mut EntityStorage,
        dirty_components: &'a RefCell<DirtyComponents>,
        context: Ctx,
    ) -> Self {
        Self {
            storage,
            dirty_components,
            modifications: RefCell::new(Modifications {
                heap: Default::default(),
                infos: Vec::with_capacity(256),
            }),
            context: RefCell::new(context),
        }
    }

    pub fn object_raw(&self, entity: &EntityId) -> Option<EntityAccess<Ctx, ()>> {
        Some(EntityAccess {
            context: &self.context,
            entry: self.storage.entry(entity)?,
            modifications: &self.modifications,
            _arch: Default::default(),
        })
    }

    pub fn object<A: StaticArchetype>(&self, entity: &EntityId) -> Option<EntityAccess<Ctx, A>> {
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
            dirty_components: self.dirty_components.borrow_mut(),
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
                dirty_components: self.dirty_components.borrow_mut(),
                entry,
            };

            unsafe { f.call(direct_access) };
        }

        mods.heap.reset();
    }
}

impl<Ctx> Drop for SceneAccess<'_, Ctx> {
    fn drop(&mut self) {
        self.apply_modifications();
    }
}

pub struct EntityAccess<'a, Ctx, A> {
    context: &'a RefCell<Ctx>,
    entry: entity_data::Entry<'a>,
    modifications: &'a RefCell<Modifications>,
    _arch: PhantomData<A>,
}

impl<'a, Ctx, A> EntityAccess<'a, Ctx, A> {
    pub fn context_mut(&self) -> RefMut<'a, Ctx> {
        self.context.borrow_mut()
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
    pub fn get<C: Component>(&self) -> &C {
        self.entry.get().unwrap()
    }

    pub fn get_mut<C: Component>(&mut self) -> &mut C {
        self.dirty_components.add::<C>(self.entry.entity());
        self.entry.get_mut().unwrap()
    }
}
