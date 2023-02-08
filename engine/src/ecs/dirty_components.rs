use std::any::TypeId;

use bitvec::vec::BitVec;
use entity_data::entity::{ArchEntityId, ArchetypeId};
use entity_data::{Component, EntityId};

use base::utils::{HashMap, HashSet};

#[derive(Default)]
pub struct DirtyComponents {
    comps_by_archetypes: Vec<HashMap<TypeId, BitVec>>,
}

impl DirtyComponents {
    pub fn add_with_component_id(&mut self, comp_id: TypeId, entity: &EntityId) {
        let arch_id = entity.archetype_id as usize;
        let id = entity.id as usize;

        if self.comps_by_archetypes.len() <= arch_id {
            self.comps_by_archetypes.resize(arch_id + 1, HashMap::new());
        }

        let components = self.comps_by_archetypes.get_mut(arch_id).unwrap();
        let bits = components.entry(comp_id).or_default();

        if bits.len() <= id {
            bits.resize(id + 1, false);
        }

        bits.set(id, true);
    }

    pub fn add<C: Component>(&mut self, entity: &EntityId) {
        self.add_with_component_id(TypeId::of::<C>(), entity);
    }

    pub fn clear_for_entity(&mut self, entity: EntityId) {
        if let Some(comps) = self.comps_by_archetypes.get_mut(entity.archetype_id as usize) {
            for bits in comps.values_mut() {
                if let Some(mut bit) = bits.get_mut(entity.id as usize) {
                    *bit = false;
                }
            }
        }
    }

    pub fn take_changes<C: Component>(&mut self) -> HashSet<EntityId> {
        let result: HashSet<_> = self
            .comps_by_archetypes
            .iter_mut()
            .enumerate()
            .flat_map(|(arch_id, components)| {
                components
                    .entry(TypeId::of::<C>())
                    .or_default()
                    .iter_ones()
                    .map(move |id| EntityId::new(arch_id as ArchetypeId, id as ArchEntityId))
            })
            .collect();

        for comps in &mut self.comps_by_archetypes {
            if let Some(bits) = comps.get_mut(&TypeId::of::<C>()) {
                bits.fill(false);
            }
        }

        result
    }
}
