#[derive(Default)]
pub struct SlotVec<T> {
    vec: Vec<Option<T>>,
    free_indices: Vec<usize>,
}

impl<T> SlotVec<T> {
    pub fn new() -> SlotVec<T> {
        SlotVec {
            vec: vec![],
            free_indices: vec![],
        }
    }

    pub fn with_capacity(capacity: usize) -> SlotVec<T> {
        SlotVec {
            vec: Vec::with_capacity(capacity),
            free_indices: Vec::with_capacity(capacity),
        }
    }

    pub fn add(&mut self, element: T) -> usize {
        if !self.free_indices.is_empty() {
            let id = self.free_indices.pop().unwrap();
            self.vec[id] = Some(element);
            id
        } else {
            self.vec.push(Some(element));
            self.vec.len() - 1
        }
    }

    pub fn remove(&mut self, id: usize) {
        self.vec[id] = None;
        self.free_indices.push(id);
    }
}
