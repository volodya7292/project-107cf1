use crate::registry::Registry;
use common::types::HashMap;
use entity_data::{EntityId, EntityStorage};
use serde::de::{DeserializeSeed, SeqAccess};
use serde::ser::{SerializeSeq, SerializeTuple};
use serde::{Deserializer, Serialize, Serializer};
use std::fmt::Formatter;

pub type StateSerializeFn = fn(
    storage: &EntityStorage,
    entity_id: &EntityId,
    serializer: &mut dyn FnMut(&dyn erased_serde::Serialize),
);

pub struct StateSerializeInfo {
    pub canon_name: &'static str,
    pub func: StateSerializeFn,
}

pub type StateDeserializeFn =
    fn(deserializer: &mut dyn erased_serde::Deserializer, storage: &mut EntityStorage) -> EntityId;

struct GenericStateValue<'a> {
    storage: &'a EntityStorage,
    entity_id: EntityId,
    state_serialize_fn: StateSerializeFn,
}

impl Serialize for GenericStateValue<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut serializer = Some(serializer);
        let mut result = None;
        (self.state_serialize_fn)(self.storage, &self.entity_id, &mut |value| {
            result = Some(value.serialize(serializer.take().unwrap()));
        });
        result.unwrap()
    }
}

struct GenericStateEntity<'a> {
    value: GenericStateValue<'a>,
    canon_name: &'static str,
}

impl Serialize for GenericStateEntity<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let arch_id = self.value.entity_id.archetype_id as u32;
        let raw_id = self.value.entity_id.id as u32;

        let mut tup = serializer.serialize_tuple(4)?;
        tup.serialize_element(self.canon_name)?;
        tup.serialize_element(&arch_id)?;
        tup.serialize_element(&raw_id)?;
        tup.serialize_element(&self.value)?;
        tup.end()
    }
}

pub struct SerializableEntityStorage<'a> {
    pub registry: &'a Registry,
    pub storage: &'a EntityStorage,
}

impl Serialize for SerializableEntityStorage<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut seq = serializer.serialize_seq(Some(self.storage.count_entities()))?;

        for entity in self.storage.entities().iter() {
            let arch = self.storage.get_archetype_by_id(entity.archetype_id).unwrap();
            let arch_ty = arch.ty();
            let ser_info = self.registry.get_block_state_serializer(arch_ty).unwrap();

            let elem = GenericStateEntity {
                value: GenericStateValue {
                    storage: self.storage,
                    entity_id: entity,
                    state_serialize_fn: ser_info.func,
                },
                canon_name: ser_info.canon_name,
            };

            SerializeSeq::serialize_element(&mut seq, &elem)?;
        }

        SerializeSeq::end(seq)
    }
}

struct StateValueDeserializer<'a> {
    storage: &'a mut EntityStorage,
    state_deserialize_fn: StateDeserializeFn,
}

impl<'de> DeserializeSeed<'de> for StateValueDeserializer<'_> {
    type Value = EntityId;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut deserializer = <dyn erased_serde::Deserializer>::erase(deserializer);
        Ok((self.state_deserialize_fn)(&mut deserializer, self.storage))
    }
}

struct DataTupleVisitor<'a> {
    registry: &'a Registry,
    storage: &'a mut EntityStorage,
}

impl<'de> serde::de::Visitor<'de> for DataTupleVisitor<'_> {
    type Value = (EntityId, EntityId);

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        write!(formatter, "a tuple (name, state)")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let canon_name = seq.next_element::<&str>()?.unwrap();
        let old_entity = {
            let arch_id = seq.next_element::<u32>()?.unwrap();
            let raw_id = seq.next_element::<u32>()?.unwrap();
            EntityId::new(arch_id, raw_id)
        };

        let state_deserialize_fn = self.registry.get_block_state_deserializer(canon_name).unwrap();
        let value_deserializer = StateValueDeserializer {
            storage: self.storage,
            state_deserialize_fn: *state_deserialize_fn,
        };
        let new_entity = seq.next_element_seed(value_deserializer)?.unwrap();

        Ok((old_entity, new_entity))
    }
}

struct DataTupleDeserializer<'a> {
    registry: &'a Registry,
    storage: &'a mut EntityStorage,
}

impl<'de> DeserializeSeed<'de> for &mut DataTupleDeserializer<'_> {
    type Value = (EntityId, EntityId);

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_tuple(
            4,
            DataTupleVisitor {
                registry: self.registry,
                storage: self.storage,
            },
        )
    }
}

struct EntityStorageVisitor<'a> {
    registry: &'a Registry,
    storage: EntityStorage,
    entity_mapping: HashMap<EntityId, EntityId>,
}

impl<'de> serde::de::Visitor<'de> for &mut EntityStorageVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
        write!(formatter, "a list of states")
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut deserializer = DataTupleDeserializer {
            registry: self.registry,
            storage: &mut self.storage,
        };

        while let Some((old_entity, new_entity)) = seq.next_element_seed(&mut deserializer)? {
            self.entity_mapping.insert(old_entity, new_entity);
        }

        Ok(())
    }
}

pub struct EntityStorageDeserializer<'a> {
    pub registry: &'a Registry,
    pub storage: EntityStorage,
    /// Maps old entity ids to new ones.
    pub entity_mapping: HashMap<EntityId, EntityId>,
}

impl<'de> DeserializeSeed<'de> for &mut EntityStorageDeserializer<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut visitor = EntityStorageVisitor {
            registry: self.registry,
            storage: Default::default(),
            entity_mapping: Default::default(),
        };
        deserializer.deserialize_seq(&mut visitor).unwrap();

        self.storage = visitor.storage;
        self.entity_mapping = visitor.entity_mapping;
        Ok(())
    }
}
