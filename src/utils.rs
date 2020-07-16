use std::mem;

pub fn is_pow_of_2(n: u64) -> bool {
    (n & (n - 1)) == 0
}

pub fn next_power_of_two(mut n: u32) -> u32 {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    n
}

pub fn make_mul_of(n: u32, m: u32) -> u32 {
    ((n + m - 1) / m) * m
}

pub fn log2(n: u32) -> u32 {
    (mem::size_of::<u32>() * 8) as u32 - n.leading_zeros() - 1
}
