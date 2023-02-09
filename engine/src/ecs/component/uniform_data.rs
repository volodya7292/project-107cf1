use crate::utils::U8SliceHelper;
use smallvec::SmallVec;
use std::mem;

pub const BASIC_UNIFORM_BLOCK_MAX_SIZE: usize = 256;

#[derive(Copy, Clone)]
pub struct UniformDataC(pub [u8; BASIC_UNIFORM_BLOCK_MAX_SIZE]);

impl UniformDataC {
    /// Safety: `T` must be annotated with `#[repr(C)]`.
    pub fn new<T: Copy>(value: T) -> Self {
        let mut raw_data = [0_u8; BASIC_UNIFORM_BLOCK_MAX_SIZE];
        raw_data[0..mem::size_of_val(&value)].raw_copy_from(value);
        Self(raw_data)
    }
}

impl Default for UniformDataC {
    fn default() -> Self {
        Self([0_u8; BASIC_UNIFORM_BLOCK_MAX_SIZE])
    }
}
