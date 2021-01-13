use crate::renderer::component;
use ahash::AHashMap;
use bit_set::BitSet;
use std::any::TypeId;
use std::collections::hash_map;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::{Arc, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

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
    fn new<T>(len: usize) -> RawComponentStorage
    where
        T: 'static + Send + Sync,
    {
        let mut vec = Vec::<MaybeUninit<T>>::new();
        vec.resize_with(len, || MaybeUninit::<T>::uninit());

        RawComponentStorage {
            data: Box::new(vec),
            available: Default::default(),
            created: Default::default(),
            modified: Default::default(),
            removed: Default::default(),
        }
    }

    /// Checks if component is present
    pub fn is_alive(&self, index: u32) -> bool {
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
            panic!("index (is {}) should be < len (is {})", index, len);
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

    pub fn alive_entries(&self) -> &BitSet {
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

    pub fn is_alive(&self, index: u32) -> bool {
        self.raw.is_alive(index)
    }

    pub fn alive_entries(&self) -> &BitSet {
        self.raw.alive_entries()
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

#[derive(Default)]
pub struct Scene {
    comp_storages: AHashMap<TypeId, Arc<RwLock<RawComponentStorage>>>,
    entity_count: u32,
    free_indices: Vec<u32>,
}

impl Scene {
    pub fn new() -> Scene {
        Default::default()
    }

    pub fn create_entity(&mut self) -> u32 {
        if self.free_indices.is_empty() {
            let index = self.entity_count;
            self.entity_count += 1;

            for comps in self.comp_storages.values() {
                let mut comps = comps.write().unwrap();
                comps.data.resize(self.entity_count as usize);
            }

            index
        } else {
            self.free_indices.pop().unwrap()
        }
    }

    pub fn remove_entities(&mut self, indices: &[u32]) {
        for comps in self.comp_storages.values() {
            let mut comps = comps.write().unwrap();

            for index in indices {
                comps.data.swap_remove(*index as usize);
            }
        }
    }

    pub fn create_renderable(
        &mut self,
        transform: component::Transform,
        renderer: component::Renderer,
        vertex_mesh: component::VertexMesh,
    ) -> u32 {
        let index = self.create_entity();
        self.storage::<component::Transform>()
            .write()
            .unwrap()
            .set(index, transform);
        self.storage::<component::Renderer>()
            .write()
            .unwrap()
            .set(index, renderer);
        self.storage::<component::VertexMesh>()
            .write()
            .unwrap()
            .set(index, vertex_mesh);
        self.storage::<component::ModelTransform>()
            .write()
            .unwrap()
            .set(index, component::ModelTransform::default());
        self.storage::<component::WorldTransform>()
            .write()
            .unwrap()
            .set(index, component::WorldTransform::default());
        index
    }

    pub fn storage<T>(&mut self) -> LockedStorage<T>
    where
        T: 'static + Send + Sync,
    {
        let raw_storage = match self.comp_storages.entry(TypeId::of::<T>()) {
            hash_map::Entry::Occupied(e) => Arc::clone(e.get()),
            hash_map::Entry::Vacant(e) => Arc::clone(e.insert(Arc::new(RwLock::new(
                RawComponentStorage::new::<T>(self.entity_count as usize),
            )))),
        };

        LockedStorage {
            raw: raw_storage,
            _ty: Default::default(),
        }
    }
}
