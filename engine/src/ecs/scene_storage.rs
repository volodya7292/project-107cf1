use crate::utils::HashMap;
use bit_set::BitSet;
use index_pool::IndexPool;
use parking_lot::{RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::any::{Any, TypeId};
use std::collections::hash_map;
use std::marker::PhantomData;
use std::sync::atomic::AtomicU32;
use std::sync::{atomic, Arc};
use std::{mem, ptr};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Entity {
    id: u32,
    gen: u64,
}

impl Entity {
    pub const NULL: Entity = Entity {
        id: u32::MAX,
        gen: u64::MAX,
    };
}

pub struct RawComponentStorage {
    data: Vec<u8>,
    elem_size: usize,
    drop_func: fn(*mut u8),
    needs_drop: bool,
    entity_allocated_count: Arc<AtomicU32>,
    emit_events: bool,
    available: BitSet,
    aval_count: u32,
    created: HashMap<u32, u64>,
    modified: HashMap<u32, u64>,
    removed: HashMap<u32, u64>,
}

#[derive(Copy, Clone)]
pub enum Event {
    // TODO: Created event proved itself unnecessary, remove it
    Created(Entity),
    Modified(Entity),
    Removed(Entity),
}

impl RawComponentStorage {
    fn new<T>(entity_allocated_count: &Arc<AtomicU32>) -> RawComponentStorage
    where
        T: 'static + Send + Sync,
    {
        let drop_func = |p: *mut u8| unsafe { ptr::drop_in_place(p as *mut T) };
        let elem_size = mem::size_of::<T>();
        let entity_count = entity_allocated_count.load(atomic::Ordering::Relaxed) as usize;

        RawComponentStorage {
            data: vec![0; entity_count * elem_size],
            elem_size,
            drop_func,
            needs_drop: mem::needs_drop::<T>(),
            entity_allocated_count: Arc::clone(entity_allocated_count),
            emit_events: false,
            available: Default::default(),
            aval_count: 0,
            created: Default::default(),
            modified: Default::default(),
            removed: Default::default(),
        }
    }

    /// Checks if a component is available.
    pub fn contains(&self, index: u32) -> bool {
        self.available.contains(index as usize)
    }

    /// Creates or modifies a component.
    ///
    /// # Safety:
    ///
    /// T must match the type of component.
    pub unsafe fn set<T>(&mut self, entity: Entity, v: T) {
        let index = entity.id as usize;
        let index_b = index * self.elem_size;

        let len = self.data.len();
        if index_b >= len {
            let new_count = self.entity_allocated_count.load(atomic::Ordering::Relaxed) as usize;
            self.data.resize(new_count * self.elem_size, 0);

            if index >= new_count {
                panic!("index (is {}) should be < len (is {})", index, new_count);
            }
        }
        let ptr = (self.data.as_mut_ptr() as *mut T).add(index);

        let already_present = !self.available.insert(index);
        if already_present {
            if self.needs_drop {
                (self.drop_func)(ptr as *mut u8);
            }
        } else {
            self.aval_count += 1;
        }

        if self.emit_events {
            if already_present {
                if !self.created.contains_key(&entity.id) {
                    self.modified.insert(entity.id, entity.gen);
                }
            } else {
                self.created.insert(entity.id, entity.gen);
            }
        }

        ptr.write(v);
    }

    /// Removes a component.
    pub fn remove(&mut self, entity: Entity) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("index (is {}) should be < len (is {})", index, len);
        }

        let index = entity.id as usize;
        let index_b = index * self.elem_size;

        let len = self.data.len();
        if index_b >= len {
            assert_failed(index, len / self.elem_size);
        }

        let was_present = self.available.remove(index);

        if self.emit_events {
            let just_created = self.created.remove(&entity.id).is_some();
            self.modified.remove(&entity.id);

            if !just_created {
                self.removed.insert(entity.id, entity.gen);
            }
        }

        if was_present {
            self.aval_count -= 1;
            if self.needs_drop {
                (self.drop_func)(unsafe { self.data.as_mut_ptr().add(index_b) });
            }
        }
    }

    /// Returns a reference to the specified component.
    ///
    /// # Safety:
    ///
    /// T must match the type of component.
    pub unsafe fn get<T>(&self, index: u32) -> Option<&T> {
        let index = index as usize;

        if self.available.contains(index) {
            Some(&*(self.data.as_ptr() as *const T).add(index))
        } else {
            None
        }
    }

    /// Modifies a component
    ///
    /// # Safety:
    ///
    /// T must match the type of component.
    pub unsafe fn get_mut<T>(&mut self, entity: Entity) -> Option<&mut T> {
        let index = entity.id as usize;

        if self.available.contains(index) {
            if self.emit_events {
                if !self.created.contains_key(&entity.id) {
                    self.modified.insert(entity.id, entity.gen);
                }
            }
            Some(&mut *(self.data.as_mut_ptr() as *mut T).add(index))
        } else {
            None
        }
    }

    /// Returns a mutable reference to the component without emitting a modification event
    pub unsafe fn get_mut_unmarked<T>(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;

        if self.available.contains(index) {
            Some(&mut *(self.data.as_mut_ptr() as *mut T).add(index))
        } else {
            None
        }
    }

    /// Modifies a component. Inserts specified value if component doesn't exist.
    ///
    /// # Safety:
    ///
    /// T must match the type of component.
    pub unsafe fn get_or_insert<T>(&mut self, entity: Entity, default: T) -> &mut T {
        let index = entity.id as usize;

        if !self.available.contains(index) {
            self.set(entity, default);
        }
        self.get_mut(entity).unwrap()
    }
}

impl Drop for RawComponentStorage {
    fn drop(&mut self) {
        if self.needs_drop {
            for index in &self.available {
                let index_b = index * self.elem_size;
                (self.drop_func)(unsafe { self.data.as_mut_ptr().add(index_b) });
            }
        }
    }
}

pub struct Entries<'a> {
    set: BitSet,
    entities: &'a Entities,
    estimate_len: u32,
}

impl Entries<'_> {
    pub fn intersection<T>(mut self, other: &impl ComponentStorageImpl<T>) -> Self {
        self.set.intersect_with(other.available());
        self.estimate_len = self.estimate_len.min(other.raw().aval_count);
        self
    }

    pub fn difference<T>(mut self, other: &impl ComponentStorageImpl<T>) -> Self {
        self.set.difference_with(other.available());
        self
    }

    pub fn iter(&self) -> EntriesIter {
        EntriesIter {
            set_iter: self.set.iter(),
            entities: self.entities,
            estimate_len: self.estimate_len,
        }
    }
}

pub struct EntriesIter<'a> {
    set_iter: bit_set::Iter<'a, u32>,
    entities: &'a Entities,
    estimate_len: u32,
}

impl Iterator for EntriesIter<'_> {
    type Item = Entity;

    fn next(&mut self) -> Option<Self::Item> {
        self.set_iter.next().map(|id| Entity {
            id: id as u32,
            gen: self.entities.generations[id],
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.estimate_len as usize, Some(self.estimate_len as usize))
    }
}

mod private {
    use crate::ecs::scene_storage::{Entities, RawComponentStorage};
    use bit_set::BitSet;

    pub trait RawComponentStorageImpl {
        fn raw(&self) -> &RawComponentStorage;
        fn entities(&self) -> &Entities;
        fn available(&self) -> &BitSet;
    }
}

pub trait ComponentStorageImpl<T>: private::RawComponentStorageImpl {
    fn len(&self) -> usize {
        self.raw().aval_count as usize
    }

    fn contains(&self, entity: Entity) -> bool {
        self.entities().is_alive(entity) && self.raw().contains(entity.id)
    }

    fn get(&self, entity: Entity) -> Option<&T> {
        if self.entities().is_alive(entity) {
            unsafe { self.raw().get::<T>(entity.id) }
        } else {
            None
        }
    }

    fn entries(&self) -> Entries {
        Entries {
            set: self.raw().available.clone(),
            entities: self.entities(),
            estimate_len: self.raw().aval_count,
        }
    }

    fn was_modified(&self, entity: Entity) -> bool {
        self.raw()
            .modified
            .get(&entity.id)
            .map(|gen| *gen == entity.gen)
            .unwrap_or(false)
            || self
                .raw()
                .created
                .get(&entity.id)
                .map(|gen| *gen == entity.gen)
                .unwrap_or(false)
    }
}

/// A immutable component storage with shared read access
pub struct ComponentStorage<'a, T> {
    raw: RwLockReadGuard<'a, RawComponentStorage>,
    entities: RwLockReadGuard<'a, Entities>,
    ty: PhantomData<T>,
}

impl<'a, T> private::RawComponentStorageImpl for ComponentStorage<'a, T> {
    fn raw(&self) -> &RawComponentStorage {
        &self.raw
    }

    fn entities(&self) -> &Entities {
        &self.entities
    }

    fn available(&self) -> &BitSet {
        &self.raw.available
    }
}

impl<'a, T> ComponentStorageImpl<T> for ComponentStorage<'a, T> {}

/// A mutable component storage with exclusive write access
pub struct ComponentStorageMut<'a, T> {
    raw: RwLockWriteGuard<'a, RawComponentStorage>,
    entities: RwLockReadGuard<'a, Entities>,
    ty: PhantomData<T>,
}

impl<'a, T> private::RawComponentStorageImpl for ComponentStorageMut<'a, T> {
    fn raw(&self) -> &RawComponentStorage {
        &self.raw
    }

    fn entities(&self) -> &Entities {
        &self.entities
    }

    fn available(&self) -> &BitSet {
        &self.raw.available
    }
}

impl<'a, T> ComponentStorageImpl<T> for ComponentStorageMut<'a, T> {}

impl<'a, T> ComponentStorageMut<'a, T> {
    #[inline]
    pub fn set(&mut self, entity: Entity, v: T) {
        if self.entities.is_alive(entity) {
            unsafe { self.raw.set(entity, v) };
        } else {
            panic!("Entity is not alive!");
        }
    }

    #[inline]
    pub fn remove(&mut self, entity: Entity) {
        if self.entities.is_alive(entity) {
            self.raw.remove(entity);
        } else {
            panic!("Entity is not alive!");
        }
    }

    #[inline]
    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        if self.entities.is_alive(entity) {
            unsafe { self.raw.get_mut::<T>(entity) }
        } else {
            None
        }
    }

    /// Inserts a new component if it doesn't exist and returns a reference to it
    #[inline]
    pub fn get_or_insert_default(&mut self, entity: Entity) -> &mut T
    where
        T: Default,
    {
        if self.entities.is_alive(entity) {
            unsafe { self.raw.get_or_insert::<T>(entity, T::default()) }
        } else {
            panic!("Entity is not alive!");
        }
    }

    /// Returns a mutable reference to the component without emitting a modification event
    #[inline]
    pub fn get_mut_unmarked(&mut self, entity: Entity) -> Option<&mut T> {
        if self.entities.is_alive(entity) {
            unsafe { self.raw.get_mut_unmarked(entity.id) }
        } else {
            None
        }
    }

    pub fn emit_events(&mut self, enable: bool) {
        self.raw.emit_events = enable;
    }

    /// Returns events and clears them internally
    pub fn events(&mut self) -> Vec<Event> {
        let mut events =
            Vec::with_capacity(self.raw.created.len() + self.raw.modified.len() + self.raw.removed.len());

        for (id, gen) in &self.raw.removed {
            events.push(Event::Removed(Entity {
                id: *id as u32,
                gen: *gen,
            }));
        }
        for (id, gen) in &self.raw.created {
            events.push(Event::Created(Entity {
                id: *id as u32,
                gen: *gen,
            }));
        }
        for (id, gen) in &self.raw.modified {
            events.push(Event::Modified(Entity {
                id: *id as u32,
                gen: *gen,
            }));
        }

        self.clear_events();
        events
    }

    pub fn clear_events(&mut self) {
        self.raw.created.clear();
        self.raw.modified.clear();
        self.raw.removed.clear();
    }
}

/// A component storage that can be locked in read or write mode
pub struct LockedStorage<'a, T> {
    raw: &'a Arc<RwLock<RawComponentStorage>>,
    entities: &'a Arc<RwLock<Entities>>,
    _ty: PhantomData<T>,
}

impl<'a, T> LockedStorage<'a, T> {
    pub fn read(&self) -> ComponentStorage<'a, T> {
        let raw = self.raw.read();
        let entities = self.entities.read();

        ComponentStorage {
            raw,
            entities,
            ty: Default::default(),
        }
    }

    pub fn write(&self) -> ComponentStorageMut<'a, T> {
        let raw = self.raw.write();
        let entities = self.entities.read();

        ComponentStorageMut {
            raw,
            entities,
            ty: Default::default(),
        }
    }
}

pub struct Entities {
    indices: IndexPool,
    entity_allocated_count: Arc<AtomicU32>,
    generations: Vec<u64>,
}

impl Entities {
    pub fn create(&mut self) -> Entity {
        assert!(self.indices.in_use() < u32::MAX as usize - 1);
        let id = self.indices.new_id();

        if id >= self.entity_allocated_count.load(atomic::Ordering::Relaxed) as usize {
            self.entity_allocated_count
                .fetch_add(1, atomic::Ordering::Relaxed);
            self.generations.push(0);
        }

        Entity {
            id: id as u32,
            gen: self.generations[id],
        }
    }

    fn free(&mut self, entity: Entity) {
        assert!(self.is_alive(entity));

        let id = entity.id as usize;
        self.indices.return_id(id).unwrap();
        self.generations[id] += 1;
    }

    pub fn is_alive(&self, entity: Entity) -> bool {
        self.generations
            .get(entity.id as usize)
            .map_or(false, |gen| *gen == entity.gen)
    }

    pub fn len(&self) -> u32 {
        self.indices.in_use() as u32
    }
}

#[derive(Default)]
pub struct Resources(HashMap<TypeId, Box<dyn Any + Send + Sync>>);

impl Resources {
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.0
            .get(&TypeId::of::<T>())
            .map(|v| v.downcast_ref::<Box<T>>().unwrap().as_ref())
    }
}

pub struct SceneStorage {
    entities: Arc<RwLock<Entities>>,
    resources: Arc<RwLock<Resources>>,
    comp_storages: HashMap<TypeId, Arc<RwLock<RawComponentStorage>>>,
    entity_allocated_count: Arc<AtomicU32>,
}

impl SceneStorage {
    pub fn new() -> SceneStorage {
        let entity_count = Arc::new(AtomicU32::new(0));

        SceneStorage {
            entities: Arc::new(RwLock::new(Entities {
                indices: Default::default(),
                entity_allocated_count: Arc::clone(&entity_count),
                generations: vec![],
            })),
            resources: Default::default(),
            comp_storages: Default::default(),
            entity_allocated_count: entity_count,
        }
    }

    pub fn add_resource<T>(&self, resource: T)
    where
        T: 'static + Send + Sync,
    {
        self.resources
            .write()
            .0
            .insert(TypeId::of::<T>(), Box::new(resource))
            .unwrap();
    }

    /// When creating multiple entities, prefer doing it through `self.entities()`.
    pub fn create_entity(&self) -> Entity {
        self.entities.write().create()
    }

    pub fn remove_entities(&self, entities: &[Entity]) {
        let mut all_entities = self.entities.write();

        for entity in entities {
            all_entities.free(*entity);
        }

        for comps in self.comp_storages.values() {
            let mut comps = comps.write();

            for &entity in entities {
                // Safety: `is_alive(entity)` check is done above in `all_entities.free`
                if comps.contains(entity.id) {
                    comps.remove(entity);
                }
            }
        }
    }

    pub fn resources(&self) -> Arc<RwLock<Resources>> {
        Arc::clone(&self.resources)
    }

    pub fn entities(&self) -> &Arc<RwLock<Entities>> {
        &self.entities
    }

    pub fn prepare_storage<T>(&mut self) -> ComponentStorageMut<T>
    where
        T: 'static + Send + Sync,
    {
        let raw_storage = match self.comp_storages.entry(TypeId::of::<T>()) {
            hash_map::Entry::Occupied(_) => panic!("Storage already prepared"),
            hash_map::Entry::Vacant(e) => e.insert(Arc::new(RwLock::new(RawComponentStorage::new::<T>(
                &self.entity_allocated_count,
            )))),
        };

        ComponentStorageMut {
            raw: raw_storage.write(),
            entities: self.entities.read(),
            ty: Default::default(),
        }
    }

    /// Get component storage without locking it
    pub fn storage<T>(&self) -> LockedStorage<T>
    where
        T: 'static + Send + Sync,
    {
        let raw_storage = &self.comp_storages[&TypeId::of::<T>()];

        LockedStorage {
            raw: raw_storage,
            entities: &self.entities,
            _ty: Default::default(),
        }
    }

    /// Get component storage locked in shared read mode
    pub fn storage_read<T>(&self) -> ComponentStorage<T>
    where
        T: 'static + Send + Sync,
    {
        let raw_storage = &self.comp_storages[&TypeId::of::<T>()];

        ComponentStorage {
            raw: raw_storage.read(),
            entities: self.entities.read(),
            ty: Default::default(),
        }
    }

    /// Get component storage locked in exclusive write mode
    pub fn storage_write<T>(&self) -> ComponentStorageMut<T>
    where
        T: 'static + Send + Sync,
    {
        let raw_storage = &self.comp_storages[&TypeId::of::<T>()];

        ComponentStorageMut {
            raw: raw_storage.write(),
            entities: self.entities.read(),
            ty: Default::default(),
        }
    }
}
