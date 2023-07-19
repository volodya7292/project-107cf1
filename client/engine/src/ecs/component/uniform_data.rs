use crate::utils::U8SliceHelper;
use common::glm::Mat4;
use std::mem;

pub(crate) const BASIC_UNIFORM_BLOCK_MAX_SIZE: usize = 256;
pub const MODEL_MATRIX_OFFSET: usize = 0;
pub const MODEL_MATRIX_SIZE: usize = mem::size_of::<Mat4>();
pub const CUSTOM_UNIFORM_BLOCK_MAX_SIZE: usize = BASIC_UNIFORM_BLOCK_MAX_SIZE - MODEL_MATRIX_SIZE;

#[derive(Copy, Clone)]
pub struct UniformDataC(pub [u8; BASIC_UNIFORM_BLOCK_MAX_SIZE]);

impl UniformDataC {
    pub fn new<T: Copy>(value: T) -> Self {
        let mut raw_data = [0_u8; BASIC_UNIFORM_BLOCK_MAX_SIZE];
        raw_data[0..mem::size_of_val(&value)].raw_copy_from(value);
        Self(raw_data)
    }

    #[inline]
    pub fn copy_from_with_offset<T: Copy>(&mut self, dst_offset: usize, value: T) {
        let offset = MODEL_MATRIX_OFFSET + MODEL_MATRIX_SIZE + dst_offset;
        self.0[offset..(offset + mem::size_of_val(&value))].raw_copy_from(value);
    }

    #[inline]
    pub fn copy_from_slice(&mut self, dst_offset: usize, slice: &[u8]) {
        let offset = MODEL_MATRIX_OFFSET + MODEL_MATRIX_SIZE + dst_offset;
        self.0[offset..(offset + slice.len())].copy_from_slice(slice);
    }

    #[inline]
    pub fn copy_from<T: Copy>(&mut self, value: T) {
        self.copy_from_with_offset(0, value)
    }
}

impl Default for UniformDataC {
    fn default() -> Self {
        Self([0_u8; BASIC_UNIFORM_BLOCK_MAX_SIZE])
    }
}
