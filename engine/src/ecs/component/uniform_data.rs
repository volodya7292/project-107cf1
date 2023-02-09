use smallvec::SmallVec;
use std::mem;

pub const BASIC_UNIFORM_BLOCK_MAX_SIZE: usize = 256;

#[derive(Copy, Clone)]
pub struct UniformDataC(pub [u8; BASIC_UNIFORM_BLOCK_MAX_SIZE]);

impl UniformDataC {
    /// Safety: `T` must be annotated with `#[repr(C)]`.
    pub fn new<T: Copy>(data: T) -> Self {
        let size = mem::size_of::<T>();
        Self::try_from(unsafe { std::slice::from_raw_parts(&data as *const _ as *const u8, size) }).unwrap()
    }
}

impl TryFrom<&[u8]> for UniformDataC {
    type Error = &'static str;

    fn try_from(data: &[u8]) -> Result<Self, Self::Error> {
        if data.len() > BASIC_UNIFORM_BLOCK_MAX_SIZE {
            return Err("data.len() must be <= BASIC_UNIFORM_BLOCK_MAX_SIZE");
        }
        let size = data.len();
        let mut raw_data = [0_u8; BASIC_UNIFORM_BLOCK_MAX_SIZE];
        raw_data[0..size].copy_from_slice(data);
        Ok(Self(raw_data))
    }
}

impl Default for UniformDataC {
    fn default() -> Self {
        Self([0_u8; BASIC_UNIFORM_BLOCK_MAX_SIZE])
    }
}
