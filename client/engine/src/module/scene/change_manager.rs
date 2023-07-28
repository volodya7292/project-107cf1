use common::types::HashMap;
use entity_data::{Component, EntityId};
use std::any::TypeId;
use std::collections::hash_map;

#[derive(Copy, Clone)]
pub struct ComponentChangesHandle {
    flow_idx: u32,
    ty: TypeId,
}

pub struct ComponentChange {
    entity: EntityId,
    ty: ChangeType,
}

impl ComponentChange {
    fn new(entity: EntityId, ty: ChangeType) -> Self {
        Self { entity, ty }
    }

    pub fn entity(&self) -> &EntityId {
        &self.entity
    }

    pub fn ty(&self) -> ChangeType {
        self.ty
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum ChangeType {
    Added,
    Modified,
    Removed,
}

impl ChangeType {
    pub fn is_new(&self) -> bool {
        matches!(*self, Self::Added | Self::Modified)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(u8)]
enum InternalChangeType {
    Added,
    Modified,
    Removed,
    RemovedThenAdded,
}

impl From<ChangeType> for InternalChangeType {
    fn from(value: ChangeType) -> Self {
        match value {
            ChangeType::Added => InternalChangeType::Added,
            ChangeType::Modified => InternalChangeType::Modified,
            ChangeType::Removed => InternalChangeType::Removed,
        }
    }
}

struct InternalComponentChange {
    ty: InternalChangeType,
    flows_mask: u64,
}

struct ComponentChanges {
    changed: HashMap<EntityId, InternalComponentChange>,
    // Max 64 flows allowed
    registered_flows_mask: u64,
}

impl ComponentChanges {
    fn register_flow(&mut self) -> u32 {
        let next_id = self.registered_flows_mask.trailing_ones();
        self.registered_flows_mask |= 1 << next_id;

        if next_id == u64::BITS {
            panic!("Maximum number {} of flows is reached!", u64::BITS);
        }
        for change in self.changed.values_mut() {
            change.flows_mask |= 1 << next_id;
        }
        next_id
    }

    fn record(&mut self, entity: EntityId, new_change: InternalChangeType) {
        let entry = self.changed.entry(entity);

        match entry {
            hash_map::Entry::Vacant(e) => {
                e.insert(InternalComponentChange {
                    ty: new_change,
                    flows_mask: self.registered_flows_mask,
                });
            }
            hash_map::Entry::Occupied(mut e) => {
                let curr_change = e.get_mut();

                match (curr_change.ty, new_change) {
                    (InternalChangeType::Added, InternalChangeType::Modified)
                    | (InternalChangeType::Modified, InternalChangeType::Modified)
                    | (InternalChangeType::RemovedThenAdded, InternalChangeType::Modified) => {
                        curr_change.flows_mask = self.registered_flows_mask;
                    }
                    (InternalChangeType::Added, InternalChangeType::Removed) => {
                        e.remove();
                    }
                    (InternalChangeType::Modified, InternalChangeType::Removed) => {
                        curr_change.ty = InternalChangeType::Removed;
                        curr_change.flows_mask = self.registered_flows_mask;
                    }
                    (InternalChangeType::Removed, InternalChangeType::Added) => {
                        curr_change.ty = InternalChangeType::RemovedThenAdded;
                        curr_change.flows_mask = self.registered_flows_mask;
                    }
                    (InternalChangeType::RemovedThenAdded, InternalChangeType::Removed) => {
                        curr_change.ty = InternalChangeType::Removed;
                        curr_change.flows_mask = self.registered_flows_mask;
                    }
                    _ => {
                        panic!(
                            "Invalid change pair: current {:?}, new {:?}",
                            curr_change.ty, new_change
                        );
                    }
                }
            }
        }
    }

    fn take(&mut self, flow_idx: u32) -> Vec<ComponentChange> {
        let flow_mask = 1 << flow_idx;
        let mut changed_entities = Vec::with_capacity(self.changed.len() * 2);

        for (entity, change) in &mut self.changed {
            if change.flows_mask & flow_mask == 0 {
                continue;
            }
            change.flows_mask &= !flow_mask;

            match change.ty {
                InternalChangeType::Added => {
                    changed_entities.push(ComponentChange::new(*entity, ChangeType::Added));
                }
                InternalChangeType::Modified => {
                    changed_entities.push(ComponentChange::new(*entity, ChangeType::Modified));
                }
                InternalChangeType::Removed => {
                    changed_entities.push(ComponentChange::new(*entity, ChangeType::Removed));
                }
                InternalChangeType::RemovedThenAdded => {
                    changed_entities.push(ComponentChange::new(*entity, ChangeType::Removed));
                    changed_entities.push(ComponentChange::new(*entity, ChangeType::Added));
                }
            }
        }

        self.changed
            .retain(|_, change| change.flows_mask.trailing_ones() > 0);

        changed_entities
    }
}

impl Default for ComponentChanges {
    fn default() -> Self {
        Self {
            changed: HashMap::with_capacity(1024),
            registered_flows_mask: 0,
        }
    }
}

pub struct SceneChangeManager {
    by_component: HashMap<TypeId, ComponentChanges>,
}

impl SceneChangeManager {
    pub(super) fn new() -> Self {
        Self {
            by_component: Default::default(),
        }
    }

    #[inline]
    fn get_component_changes(&mut self, ty: TypeId) -> &mut ComponentChanges {
        self.by_component.entry(ty).or_default()
    }

    /// Registers new change receiver for the specified component id.
    pub fn register_component_flow_for_id(&mut self, comp_id: TypeId) -> ComponentChangesHandle {
        let changes = self.get_component_changes(comp_id);
        let new_flow_idx = changes.register_flow();

        ComponentChangesHandle {
            flow_idx: new_flow_idx,
            ty: comp_id,
        }
    }

    /// Registers new change receiver for component `C`.
    pub fn register_component_flow<C: Component>(&mut self) -> ComponentChangesHandle {
        self.register_component_flow_for_id(TypeId::of::<C>())
    }

    /// Marks component of type `ty` of `entity` with `change` mark.
    #[inline]
    pub(super) fn record(&mut self, ty: TypeId, entity: EntityId, change: ChangeType) {
        self.get_component_changes(ty).record(entity, change.into())
    }

    /// Marks component of type `ty` of `entity` as modified.
    #[inline]
    pub fn record_modification_by_component_id(&mut self, ty: TypeId, entity: EntityId) {
        self.record(ty, entity, ChangeType::Modified);
    }

    /// Marks component `C` of `entity` as modified.
    #[inline]
    pub fn record_modification<C: Component>(&mut self, entity: EntityId) {
        self.record_modification_by_component_id(TypeId::of::<C>(), entity);
    }

    /// Removes all change-marks from `handle` and returns the changes.
    #[inline]
    pub fn take(&mut self, handle: ComponentChangesHandle) -> Vec<ComponentChange> {
        self.get_component_changes(handle.ty).take(handle.flow_idx)
    }

    /// Removes all change-marks from `handle` and returns only new-value changes.
    /// That is, the components that have new value: added or modified.
    #[inline]
    pub fn take_new_iter(&mut self, handle: ComponentChangesHandle) -> impl Iterator<Item = EntityId> {
        self.take(handle)
            .into_iter()
            .filter_map(|v| v.ty.is_new().then_some(v.entity))
    }

    /// Removes all change-marks from `handle` and returns only new-value changes.
    /// That is, the components that have new value: added or modified.
    #[inline]
    pub fn take_new<R: FromIterator<EntityId>>(&mut self, handle: ComponentChangesHandle) -> R {
        self.take(handle)
            .into_iter()
            .filter_map(|v| v.ty.is_new().then_some(v.entity))
            .collect()
    }
}
