use smallvec::SmallVec;

pub const BASIC_UNIFORM_BLOCK_MAX_SIZE: usize = 256;

pub struct UniformData([u8; BASIC_UNIFORM_BLOCK_MAX_SIZE]);
