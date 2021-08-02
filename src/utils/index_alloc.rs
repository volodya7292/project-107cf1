use bit_set::BitSet;
use bit_vec::BitVec;

pub struct IndexAlloc {
    free_indices: BitSet,
    size: usize,
}

impl IndexAlloc {
    pub fn new(size: usize) -> IndexAlloc {
        IndexAlloc {
            free_indices: BitSet::from_bit_vec(BitVec::from_elem(size, true)),
            size,
        }
    }

    pub fn alloc(&mut self) -> Option<usize> {
        let id = self.free_indices.iter().next()?;
        self.free_indices.remove(id);
        Some(id)
    }

    pub fn free(&mut self, index: usize) {
        #[cold]
        #[inline(never)]
        fn assert_failed(index: usize, size: usize) -> ! {
            panic!("index (is {}) should be < size (is {})", index, size);
        }

        if index >= self.size {
            assert_failed(index, self.size);
        }
        self.free_indices.insert(index);
    }
}
