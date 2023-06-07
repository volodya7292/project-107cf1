use index_pool::IndexPool;

#[derive(Default)]
pub struct IndexAllocator {
    pool: IndexPool,
}

impl IndexAllocator {
    pub fn new() -> Self {
        Self {
            pool: Default::default(),
        }
    }

    pub fn alloc(&mut self) -> usize {
        self.pool.new_id()
    }

    /// Returns `true` if the element was present.
    pub fn free(&mut self, index: usize) -> bool {
        let result = self.pool.return_id(index);
        result.is_ok()
    }

    pub fn contains(&self, index: usize) -> bool {
        !self.pool.is_free(index)
    }

    pub fn clear(&mut self) {
        self.pool.clear();
    }

    pub fn iter(&self) -> Iter {
        Iter(self.pool.all_indices())
    }
}

pub struct Iter<'a>(index_pool::iter::IndexIter<'a>);

impl Iterator for Iter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}
