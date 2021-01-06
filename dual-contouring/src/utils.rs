use std::mem;

pub fn is_pow_of_2(n: u64) -> bool {
    (n & (n - 1)) == 0
}

pub fn log2(n: u32) -> u32 {
    (mem::size_of::<u32>() * 8) as u32 - n.leading_zeros() - 1
}
