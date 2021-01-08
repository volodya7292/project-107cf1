use crate::renderer::component;
use bit_set::BitSet;
use std::sync::{Arc, RwLock};

pub struct ComponentStorage<T> {
    data: Vec<Option<T>>,
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

impl<T> ComponentStorage<T> {
    fn resize(&mut self, new_len: u32) {
        self.data.resize_with(new_len as usize, || None);
    }

    pub fn is_alive(&self, index: u32) -> bool {
        self.data
            .get(index as usize)
            .map(|e| e.is_some())
            .unwrap_or(false)
    }

    /// Creates or modifies component
    pub fn set(&mut self, index: u32, v: T) {
        let index = index as usize;
        let comp = &mut self.data[index];

        if comp.is_some() {
            if !self.created.contains(index) {
                self.modified.insert(index);
            }
        } else {
            if self.removed.contains(index) {
                self.removed.remove(index);
                self.modified.insert(index);
            } else {
                self.created.insert(index);
            }
        }

        *comp = Some(v);
    }

    fn remove(&mut self, index: u32) {
        let index = index as usize;
        self.data[index] = None;

        if self.created.contains(index) {
            self.created.remove(index);
        } else {
            self.removed.insert(index);
        }
        self.modified.remove(index);
    }

    pub fn get(&self, index: u32) -> Option<&T> {
        self.data.get(index as usize).map(|e| e.as_ref()).flatten()
    }

    /// Modifies component
    pub fn get_mut(&mut self, index: u32) -> Option<&mut T> {
        let index = index as usize;
        self.created.remove(index);
        self.modified.insert(index);
        self.data.get_mut(index).map(|e| e.as_mut()).flatten()
    }

    /// Get mutable component without emission modification event
    pub fn get_mut_unchecked(&mut self, index: u32) -> Option<&mut T> {
        self.data.get_mut(index as usize).map(|e| e.as_mut()).flatten()
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

impl<T> Default for ComponentStorage<T> {
    fn default() -> Self {
        ComponentStorage {
            data: vec![],
            created: Default::default(),
            modified: Default::default(),
            removed: Default::default(),
        }
    }
}

#[derive(Default)]
pub struct Scene {
    transform_comps: Arc<RwLock<ComponentStorage<component::Transform>>>,
    model_transform_comps: Arc<RwLock<ComponentStorage<component::ModelTransform>>>,
    world_transform_comps: Arc<RwLock<ComponentStorage<component::WorldTransform>>>,
    renderer_comps: Arc<RwLock<ComponentStorage<component::Renderer>>>,
    vertex_mesh_comps: Arc<RwLock<ComponentStorage<component::VertexMesh>>>,
    camera_comps: Arc<RwLock<ComponentStorage<component::Camera>>>,

    entity_count: u32,
    free_indices: Vec<u32>,

    renderables: Vec<u32>,
}

impl Scene {
    pub fn new() -> Scene {
        Default::default()
    }

    pub fn create_entity(&mut self) -> u32 {
        if self.free_indices.is_empty() {
            let index = self.entity_count;
            self.entity_count += 1;

            self.transform_comps.write().unwrap().resize(self.entity_count);
            self.model_transform_comps
                .write()
                .unwrap()
                .resize(self.entity_count);
            self.world_transform_comps
                .write()
                .unwrap()
                .resize(self.entity_count);
            self.renderer_comps.write().unwrap().resize(self.entity_count);
            self.vertex_mesh_comps.write().unwrap().resize(self.entity_count);
            self.camera_comps.write().unwrap().resize(self.entity_count);

            index
        } else {
            self.free_indices.pop().unwrap()
        }
    }

    pub fn create_renderable(
        &mut self,
        transform: component::Transform,
        renderer: component::Renderer,
        vertex_mesh: component::VertexMesh,
    ) -> u32 {
        let index = self.create_entity();
        self.transform_comps.write().unwrap().set(index, transform);
        self.renderer_comps.write().unwrap().set(index, renderer);
        self.vertex_mesh_comps.write().unwrap().set(index, vertex_mesh);
        self.renderables.push(index);
        index
    }

    pub fn remove_renderables(&mut self, entities: &[u32]) {
        let mut transform_comps = self.transform_comps.write().unwrap();
        let mut renderer_comps = self.renderer_comps.write().unwrap();
        let mut vertex_mesh_comps = self.vertex_mesh_comps.write().unwrap();

        for &index in entities {
            transform_comps.remove(index);
            renderer_comps.remove(index);
            vertex_mesh_comps.remove(index);
        }
    }

    pub fn renderables(&self) -> &[u32] {
        &self.renderables
    }

    pub fn transform_components(&self) -> Arc<RwLock<ComponentStorage<component::Transform>>> {
        Arc::clone(&self.transform_comps)
    }

    pub fn renderer_components(&self) -> Arc<RwLock<ComponentStorage<component::Renderer>>> {
        Arc::clone(&self.renderer_comps)
    }

    pub fn vertex_mesh_components(&self) -> Arc<RwLock<ComponentStorage<component::VertexMesh>>> {
        Arc::clone(&self.vertex_mesh_comps)
    }

    pub fn camera_components(&self) -> Arc<RwLock<ComponentStorage<component::Camera>>> {
        Arc::clone(&self.camera_comps)
    }
}
