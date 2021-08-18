use crate::utils::HashMap;
use bit_set::BitSet;
use index_pool::IndexPool;
use std::any::{Any, TypeId};
use std::collections::hash_map;
use std::marker::PhantomData;
use std::sync::atomic::AtomicU32;
use std::sync::{atomic, Arc, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::{mem, ptr};

#[derive(Copy, Clone, Eq, PartialEq, Hash)]
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
                if let hash_map::Entry::Vacant(e) = self.removed.entry(entity.id) {
                    e.insert(entity.gen);
                }
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
    use crate::render_engine::scene::{Entities, RawComponentStorage};
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
}

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

        self.raw.created.clear();
        self.raw.modified.clear();
        self.raw.removed.clear();

        events
    }
}

pub struct LockedStorage<T> {
    raw: Arc<RwLock<RawComponentStorage>>,
    entities: Arc<RwLock<Entities>>,
    _ty: PhantomData<T>,
}

impl<T> LockedStorage<T> {
    pub fn read(&self) -> ComponentStorage<T> {
        let raw = self.raw.read().unwrap();
        let entities = self.entities.read().unwrap();

        ComponentStorage {
            raw,
            entities,
            ty: Default::default(),
        }
    }

    pub fn write(&self) -> ComponentStorageMut<T> {
        let raw = self.raw.write().unwrap();
        let entities = self.entities.read().unwrap();

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
        self.generations[id] += 1;

        Entity {
            id: id as u32,
            gen: self.generations[id],
        }
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

pub struct Scene {
    entities: Arc<RwLock<Entities>>,
    resources: Arc<RwLock<Resources>>,
    comp_storages: Mutex<HashMap<TypeId, Arc<RwLock<RawComponentStorage>>>>,
    entity_allocated_count: Arc<AtomicU32>,
}

impl Scene {
    pub fn new() -> Scene {
        let entity_count = Arc::new(AtomicU32::new(0));

        Scene {
            entities: Arc::new(RwLock::new(Entities {
                indices: Default::default(),
                entity_allocated_count: Arc::clone(&entity_count),
                generations: vec![],
            })),
            resources: Default::default(),
            comp_storages: Mutex::new(Default::default()),
            entity_allocated_count: entity_count,
        }
    }

    pub fn add_resource<T>(&self, resource: T)
    where
        T: 'static + Send + Sync,
    {
        self.resources
            .write()
            .unwrap()
            .0
            .insert(TypeId::of::<T>(), Box::new(resource))
            .unwrap();
    }

    /// When creating multiple entities, prefer doing it through `self.entities()`.
    pub fn create_entity(&self) -> Entity {
        self.entities.write().unwrap().create()
    }

    pub fn remove_entities(&self, entities: &[Entity]) {
        let mut all_entities = self.entities.write().unwrap();

        for entity in entities {
            if all_entities.is_alive(*entity) {
                all_entities.indices.return_id(entity.id as usize).unwrap();
            } else {
                panic!("Entity is not alive!");
            }
        }

        for comps in self.comp_storages.lock().unwrap().values() {
            let mut comps = comps.write().unwrap();

            for &entity in entities {
                if all_entities.is_alive(entity) {
                    if comps.contains(entity.id) {
                        comps.remove(entity);
                    }
                } else {
                    panic!("Entity is not alive!");
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

    pub fn storage<T>(&self) -> LockedStorage<T>
    where
        T: 'static + Send + Sync,
    {
        let raw_storage = match self.comp_storages.lock().unwrap().entry(TypeId::of::<T>()) {
            hash_map::Entry::Occupied(e) => Arc::clone(e.get()),
            hash_map::Entry::Vacant(e) => Arc::clone(e.insert(Arc::new(RwLock::new(
                RawComponentStorage::new::<T>(&self.entity_allocated_count),
            )))),
        };

        LockedStorage {
            raw: raw_storage,
            entities: Arc::clone(&self.entities),
            _ty: Default::default(),
        }
    }
}
