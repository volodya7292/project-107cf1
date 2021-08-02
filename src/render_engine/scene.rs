use crate::utils::HashMap;
use bit_set::BitSet;
use std::any::{Any, TypeId};
use std::collections::hash_map;
use std::marker::PhantomData;
use std::sync::atomic::AtomicU32;
use std::sync::{atomic, Arc, Mutex, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::{mem, ptr};

pub struct RawComponentStorage {
    data: Vec<u8>,
    elem_size: usize,
    drop_func: fn(*mut u8),
    needs_drop: bool,
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
        let drop_func = |p: *mut u8| unsafe { ptr::drop_in_place(p as *mut T) };
        let elem_size = mem::size_of::<T>();
        let entity_count = entity_allocated_count.load(atomic::Ordering::Relaxed) as usize;

        RawComponentStorage {
            data: vec![0; entity_count * elem_size],
            elem_size,
            drop_func,
            needs_drop: mem::needs_drop::<T>(),
            entity_allocated_count: Arc::clone(entity_allocated_count),
            available: Default::default(),
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
    pub unsafe fn set<T>(&mut self, index: u32, v: T) {
        let index = index as usize;
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
            if !self.created.contains(index) {
                self.modified.insert(index);
            }
        } else {
            let was_removed = self.removed.remove(index);
            if was_removed {
                self.modified.insert(index);
            } else {
                self.created.insert(index);
            }
        }

        ptr.write(v);
    }

    /// Removes a component.
    pub fn remove(&mut self, index: u32) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, len: usize) -> ! {
            panic!("index (is {}) should be < len (is {})", index, len);
        }

        let index = index as usize;
        let index_b = index * self.elem_size;

        let len = self.data.len();
        if index_b >= len {
            assert_failed(index, len / self.elem_size);
        }

        let was_present = self.available.remove(index);
        let just_created = self.created.remove(index);
        self.modified.remove(index);
        if !just_created {
            self.removed.insert(index);
        }

        if was_present && self.needs_drop {
            (self.drop_func)(unsafe { self.data.as_mut_ptr().add(index_b) });
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
    pub unsafe fn get_mut<T>(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;

        if self.available.contains(index) {
            if !self.created.contains(index) {
                self.modified.insert(index);
            }
            Some(&mut *(self.data.as_mut_ptr() as *mut T).add(index))
        } else {
            None
        }
    }

    /// Returns a mutable reference to the component without emitting a modification event
    pub unsafe fn get_mut_unchecked<T>(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;

        if self.available.contains(index) {
            Some(&mut *(self.data.as_mut_ptr() as *mut T).add(index))
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

        for index in &self.removed {
            events.push(Event::Removed(index as u32));
        }
        for index in &self.created {
            events.push(Event::Created(index as u32));
        }
        for index in &self.modified {
            events.push(Event::Modified(index as u32));
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
    pub fn contains(&self, index: u32) -> bool {
        self.raw.contains(index)
    }

    pub fn get(&self, index: u32) -> Option<&T> {
        unsafe { self.raw.get::<T>(index) }
    }

    pub fn entries(&self) -> &BitSet {
        self.raw.entries()
    }
}

pub struct ComponentStorageMut<'a, T> {
    raw: RwLockWriteGuard<'a, RawComponentStorage>,
    ty: PhantomData<T>,
}

impl<'a, T> ComponentStorageMut<'a, T> {
    #[inline]
    pub fn set(&mut self, index: u32, v: T) {
        unsafe { self.raw.set(index, v) };
    }

    #[inline]
    pub fn remove(&mut self, index: u32) {
        self.raw.remove(index);
    }

    #[inline]
    pub fn get(&self, index: u32) -> Option<&T> {
        unsafe { self.raw.get::<T>(index) }
    }

    #[inline]
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        unsafe { self.raw.get_mut::<T>(index) }
    }

    #[inline]
    pub fn get_mut_unchecked(&mut self, index: u32) -> Option<&mut T> {
        unsafe { self.raw.get_mut_unchecked(index) }
    }

    #[inline]
    pub fn events(&mut self) -> Vec<Event> {
        self.raw.events()
    }

    #[inline]
    pub fn contains(&self, index: u32) -> bool {
        self.raw.contains(index)
    }

    #[inline]
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
                .fetch_add(1, atomic::Ordering::Relaxed) as u32
        }
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
    entities: Arc<Mutex<Entities>>,
    entity_allocated_count: Arc<AtomicU32>,
    resources: Arc<RwLock<Resources>>,
    comp_storages: Mutex<HashMap<TypeId, Arc<RwLock<RawComponentStorage>>>>,
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

            for &index in indices {
                if comps.contains(index) {
                    comps.remove(index);
                }
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
