use bit_set::BitSet;

#[derive(Default)]
pub struct SlotVec<T> {
    vec: Vec<Option<T>>,
    free_indices: BitSet,
}

impl<T> SlotVec<T> {
    pub fn new() -> SlotVec<T> {
        SlotVec {
            vec: vec![],
            free_indices: BitSet::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> SlotVec<T> {
        SlotVec {
            vec: Vec::with_capacity(capacity),
            free_indices: BitSet::with_capacity(capacity),
        }
    }

    /// Returns index of added element.
    pub fn add(&mut self, element: T) -> usize {
        if let Some(id) = self.free_indices.iter().next() {
            self.free_indices.remove(id);
            self.vec[id] = Some(element);
            id
        } else {
            self.vec.push(Some(element));
            self.vec.len() - 1
        }
    }

    pub fn remove(&mut self, id: usize) {
        self.vec[id] = None;
        self.free_indices.insert(id);
    }
}
