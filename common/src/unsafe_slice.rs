use std::marker::PhantomData;
use std::ops::Range;
use std::slice;

#[derive(Copy, Clone)]
pub struct UnsafeSlice<'a, T> {
    ptr: *mut T,
    len: usize,
    _lifetime: PhantomData<&'a ()>,
}

unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _lifetime: Default::default(),
        }
    }

    /// # Safety
    /// It is UB if two threads write to the same index without synchronization.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut(&self, range: Range<usize>) -> &mut [T] {
        unsafe {
            assert!(range.end <= self.len);
            slice::from_raw_parts_mut(self.ptr.add(range.start), range.len())
        }
    }
}
