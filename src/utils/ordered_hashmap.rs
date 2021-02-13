use crate::utils::HashMap;
use std::collections::hash_map;
use std::hash::{BuildHasher, Hash};
use std::iter::FusedIterator;
use std::ops::Range;
use std::slice;

#[derive(Clone)]
struct Element<K, V> {
    key: K,
    value: V,
}

#[derive(Clone)]
pub struct OrderedHashMap<K, V> {
    vec: Vec<Element<K, V>>,
    map: HashMap<K, usize>,
}

impl<K, V> OrderedHashMap<K, V>
where
    K: Clone + Eq + Hash,
{
    pub fn new() -> OrderedHashMap<K, V> {
        Default::default()
    }

    fn update_map(&mut self, range: Range<usize>) {
        let map = &mut self.map;

        self.vec[range].iter().enumerate().for_each(|(i, e)| {
            *map.get_mut(&e.key).unwrap() = i;
        });
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn insert(&mut self, key: K, value: V) {
        use hash_map::Entry;

        let element = Element {
            key: key.clone(),
            value,
        };

        match self.map.entry(key.clone()) {
            Entry::Occupied(e) => {
                self.vec[*e.get()] = element;
            }
            Entry::Vacant(e) => {
                self.vec.push(element);
                e.insert(self.vec.len() - 1);
            }
        }
    }

    pub fn swap_remove(&mut self, key: &K) -> Option<V> {
        let i = self.map.remove(key)?;
        let value = self.vec.swap_remove(i).value;

        if let Some(e) = self.vec.get(i) {
            // Update position of the swapped element
            *self.map.get_mut(&e.key).unwrap() = i;
        }

        Some(value)
    }

    pub fn get_at(&self, index: usize) -> Option<&V> {
        self.vec.get(index).map(|e| &e.value)
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.vec.get(*self.map.get(key)?).map(|e| &e.value)
    }

    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.vec.get_mut(*self.map.get(key)?).map(|e| &mut e.value)
    }

    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            base: self.vec.iter(),
        }
    }

    pub fn iter_mut(&mut self) -> IterMut<K, V> {
        IterMut {
            base: self.vec.iter_mut(),
        }
    }
}

impl<K, V> Default for OrderedHashMap<K, V> {
    fn default() -> OrderedHashMap<K, V> {
        OrderedHashMap {
            vec: vec![],
            map: HashMap::with_hasher(Default::default()),
        }
    }
}

pub struct Iter<'a, K, V> {
    base: slice::Iter<'a, Element<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|e| (&e.key, &e.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.base.nth(n).map(|e| (&e.key, &e.value))
    }
}

impl<'a, K, V> FusedIterator for Iter<'a, K, V> {}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize {
        self.base.len()
    }
}

pub struct IterMut<'a, K, V> {
    base: slice::IterMut<'a, Element<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.base.next().map(|e| (&e.key, &mut e.value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.base.size_hint()
    }

    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.base.nth(n).map(|e| (&e.key, &mut e.value))
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {
    fn len(&self) -> usize {
        self.base.len()
    }
}

impl<'a, K, V> FusedIterator for IterMut<'a, K, V> {}
