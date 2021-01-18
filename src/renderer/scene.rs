use ahash::AHashMap;
use bit_set::BitSet;
use std::any::{Any, TypeId};
use std::collections::hash_map;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicU32;
use std::sync::{atomic, Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

trait Storage: Send + Sync {
    fn len(&self) -> usize;
    fn resize(&mut self, new_len: usize);
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
    fn swap_remove(&mut self, index: usize);
}

impl<T> Storage for Vec<MaybeUninit<T>>
where
    T: Send + Sync,
{
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn resize(&mut self, new_len: usize) {
        Vec::resize_with(self, new_len, || MaybeUninit::<T>::uninit());
    }

    fn as_ptr(&self) -> *const u8 {
        Vec::as_ptr(self) as *const u8
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        Vec::as_mut_ptr(self) as *mut u8
    }

    fn swap_remove(&mut self, index: usize) {
        Vec::swap_remove(self, index);
    }
}

pub struct RawComponentStorage {
    data: Box<dyn Storage>,
    entity_allocated_count: Arc<AtomicU32>,
    available: BitSet,
    created: BitSet,
    modified: BitSet,
    removed: BitSet,
}

#[derive(Copy, Clone)]
pub enum Event {
    Created(u32),
    Modified(u32),
    Removed(u32),
}

impl RawComponentStorage {
    fn new<T>(entity_allocated_count: &Arc<AtomicU32>) -> RawComponentStorage
    where
        T: 'static + Send + Sync,
    {
        RawComponentStorage {
            data: Box::new(Vec::<MaybeUninit<T>>::new()),
            entity_allocated_count: Arc::clone(entity_allocated_count),
            available: Default::default(),
            created: Default::default(),
            modified: Default::default(),
            removed: Default::default(),
        }
    }

    /// Checks if component is present
    pub fn contains(&self, index: u32) -> bool {
        self.available.contains(index as usize)
    }

    /// Creates or modifies component
    ///
    /// # Safety
    /// size of T must equal self.type_size
    pub unsafe fn set<T>(&mut self, index: u32, v: T) {
        let index = index as usize;

        let len = self.data.len();
        if index >= len {
            let new_len = self.entity_allocated_count.load(atomic::Ordering::Relaxed) as usize;
            self.data.resize(new_len);

            if index >= new_len {
                panic!("index (is {}) should be < len (is {})", index, new_len);
            }
        }

        let val = &mut *(self.data.as_mut_ptr() as *mut MaybeUninit<T>).offset(index as isize);

        if !self.available.insert(index) {
            val.as_mut_ptr().drop_in_place();

            if !self.created.contains(index) {
                self.modified.insert(index);
            }
        } else {
            if self.removed.remove(index) {
                self.modified.insert(index);
            } else {
                self.created.insert(index);
            }
        }

        val.as_mut_ptr().write(v);
    }

    /// # Safety
    /// size of T must equal self.type_size
    unsafe fn remove<T>(&mut self, index: u32) {
        let index = index as usize;

        let len = self.data.len();
        if index >= len {
            panic!("index (is {}) should be < len (is {})", index, len);
        }

        let val = &mut *(self.data.as_mut_ptr() as *mut MaybeUninit<T>).offset(index as isize);

        if self.available.remove(index) {
            val.as_mut_ptr().drop_in_place();
        }

        if !self.created.remove(index) {
            self.removed.insert(index);
        }
        self.modified.remove(index);
    }

    /// # Safety
    /// size of T must equal self.type_size
    pub unsafe fn get<T>(&self, index: u32) -> Option<&T> {
        let index = index as usize;

        if self.available.contains(index) {
            Some(&*(&*(self.data.as_ptr() as *const MaybeUninit<T>).offset(index as isize)).as_ptr())
        } else {
            None
        }
    }

    /// Modifies component
    ///
    /// # Safety
    /// size of T must equal self.type_size
    pub unsafe fn get_mut<T>(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;

        if self.available.contains(index) {
            self.created.remove(index);
            self.modified.insert(index);

            Some(
                &mut *(&mut *(self.data.as_mut_ptr() as *mut MaybeUninit<T>).offset(index as isize))
                    .as_mut_ptr(),
            )
        } else {
            None
        }
    }

    /// Get mutable component without emission modification event
    ///
    /// # Safety
    /// size of T must equal self.type_size
    pub unsafe fn get_mut_unchecked<T>(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;

        if self.available.contains(index) {
            Some(
                &mut *(&mut *(self.data.as_mut_ptr() as *mut MaybeUninit<T>).offset(index as isize))
                    .as_mut_ptr(),
            )
        } else {
            None
        }
    }

    pub fn entries(&self) -> &BitSet {
        &self.available
    }

    /// Returns events and clears them internally
    pub fn events(&mut self) -> Vec<Event> {
        let mut events = Vec::with_capacity(self.created.len() + self.modified.len() + self.removed.len());

        for index in &self.created {
            events.push(Event::Created(index as u32));
        }
        for index in &self.modified {
            events.push(Event::Modified(index as u32));
        }
        for index in &self.removed {
            events.push(Event::Removed(index as u32));
        }

        self.created.clear();
        self.modified.clear();
        self.removed.clear();

        events
    }
}

pub struct ComponentStorage<'a, T> {
    raw: RwLockReadGuard<'a, RawComponentStorage>,
    ty: PhantomData<T>,
}

impl<'a, T> ComponentStorage<'a, T> {
    pub fn get(&self, index: u32) -> Option<&T> {
        unsafe { self.raw.get::<T>(index) }
    }
}

pub struct ComponentStorageMut<'a, T> {
    raw: RwLockWriteGuard<'a, RawComponentStorage>,
    ty: PhantomData<T>,
}

impl<'a, T> ComponentStorageMut<'a, T> {
    pub fn set(&mut self, index: u32, v: T) {
        unsafe { self.raw.set(index, v) };
    }

    pub fn remove(&mut self, index: u32) {
        unsafe { self.raw.remove::<T>(index) };
    }

    pub fn get(&self, index: u32) -> Option<&T> {
        unsafe { self.raw.get::<T>(index) }
    }

    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        unsafe { self.raw.get_mut::<T>(index) }
    }

    pub fn get_mut_unchecked(&mut self, index: u32) -> Option<&mut T> {
        unsafe { self.raw.get_mut_unchecked(index) }
    }

    pub fn events(&mut self) -> Vec<Event> {
        self.raw.events()
    }

    pub fn contains(&self, index: u32) -> bool {
        self.raw.contains(index)
    }

    pub fn entries(&self) -> &BitSet {
        self.raw.entries()
    }
}

pub struct LockedStorage<T> {
    raw: Arc<RwLock<RawComponentStorage>>,
    _ty: PhantomData<T>,
}

impl<T> LockedStorage<T> {
    pub fn read(&self) -> Result<ComponentStorage<T>, PoisonError<RwLockReadGuard<RawComponentStorage>>> {
        self.raw.read().map(|v| ComponentStorage {
            raw: v,
            ty: Default::default(),
        })
    }

    pub fn write(
        &self,
    ) -> Result<ComponentStorageMut<T>, PoisonError<RwLockWriteGuard<RawComponentStorage>>> {
        self.raw.write().map(|v| ComponentStorageMut {
            raw: v,
            ty: Default::default(),
        })
    }
}

pub struct Entities {
    free_indices: BitSet,
    entity_allocated_count: Arc<AtomicU32>,
}

impl Entities {
    pub fn create(&mut self) -> u32 {
        if let Some(index) = self.free_indices.iter().next() {
            self.free_indices.remove(index);
            index as u32
        } else {
            self.entity_allocated_count
                .fetch_add(1, atomic::Ordering::Relaxed)
        }
    }
}

#[derive(Default)]
pub struct Resources(AHashMap<TypeId, Box<dyn Any + Send + Sync>>);

impl Resources {
    pub fn get<T: 'static>(&self) -> Option<&T> {
        self.0
            .get(&TypeId::of::<T>())
            .map(|v| v.downcast_ref::<Box<T>>().unwrap().as_ref())
    }
}

pub struct Scene {
    entities: Arc<Mutex<Entities>>,
    entity_allocated_count: Arc<AtomicU32>,
    resources: Arc<RwLock<Resources>>,
    comp_storages: Mutex<AHashMap<TypeId, Arc<RwLock<RawComponentStorage>>>>,
}

impl Scene {
    pub fn new() -> Scene {
        let entity_count = Arc::new(AtomicU32::new(0));

        Scene {
            entities: Arc::new(Mutex::new(Entities {
                free_indices: Default::default(),
                entity_allocated_count: Arc::clone(&entity_count),
            })),
            entity_allocated_count: entity_count,
            resources: Default::default(),
            comp_storages: Mutex::new(Default::default()),
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

    pub fn create_entity(&self) -> u32 {
        self.entities.lock().unwrap().create()
    }

    pub fn remove_entities(&self, indices: &[u32]) {
        self.entities
            .lock()
            .unwrap()
            .free_indices
            .extend(indices.iter().map(|&v| v as usize));

        for comps in self.comp_storages.lock().unwrap().values() {
            let mut comps = comps.write().unwrap();

            for index in indices {
                comps.data.swap_remove(*index as usize);
            }
        }
    }

    pub fn resources(&self) -> Arc<RwLock<Resources>> {
        Arc::clone(&self.resources)
    }

    pub fn entities(&self) -> &Arc<Mutex<Entities>> {
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
            _ty: Default::default(),
        }
    }
}
