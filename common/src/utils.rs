use nalgebra_glm as glm;
use nalgebra_glm::{RealNumber, Vec3};
use std::time::{Duration, Instant};
use std::{mem, slice, thread};

pub const fn prev_power_of_two(mut n: u32) -> u32 {
    n = n | (n >> 1);
    n = n | (n >> 2);
    n = n | (n >> 4);
    n = n | (n >> 8);
    n = n | (n >> 16);
    n - (n >> 1)
}

pub trait AllSame {
    fn all_same(&mut self) -> bool;
}

pub trait AllSameBy<I: Iterator> {
    fn all_same_by<F>(&mut self, f: F) -> bool
    where
        F: FnMut(&I::Item, &I::Item) -> bool;

    fn all_same_by_key<F, K>(&mut self, f: F) -> bool
    where
        F: FnMut(&I::Item) -> K,
        K: PartialEq;
}

impl<I: Iterator> AllSame for I
where
    I::Item: PartialEq,
{
    fn all_same(&mut self) -> bool {
        if let Some(first) = self.next() {
            self.all(|v| v == first)
        } else {
            true
        }
    }
}

impl<I: Iterator> AllSameBy<I> for I {
    fn all_same_by<F>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&I::Item, &I::Item) -> bool,
    {
        if let Some(first) = self.next() {
            self.all(|v| f(&first, &v))
        } else {
            true
        }
    }

    fn all_same_by_key<F, K>(&mut self, mut f: F) -> bool
    where
        F: FnMut(&I::Item) -> K,
        K: PartialEq,
    {
        if let Some(first) = self.next() {
            let first = f(&first);
            self.all(|v| f(&v) == first)
        } else {
            true
        }
    }
}

pub trait StringExt {
    fn insert_at_char(&mut self, char_idx: usize, ch: char);
    fn insert_str_at_char(&mut self, char_idx: usize, s: &str);
    fn remove_at_char(&mut self, char_idx: usize);
}

impl StringExt for String {
    fn insert_at_char(&mut self, char_idx: usize, ch: char) {
        let byte_idx = self
            .char_indices()
            .skip(char_idx)
            .next()
            .map(|v| v.0)
            .unwrap_or(self.len());
        self.insert(byte_idx, ch);
    }

    fn insert_str_at_char(&mut self, char_idx: usize, s: &str) {
        let byte_idx = self
            .char_indices()
            .skip(char_idx)
            .next()
            .map(|v| v.0)
            .unwrap_or(self.len());
        self.insert_str(byte_idx, s);
    }

    fn remove_at_char(&mut self, char_idx: usize) {
        let (byte_idx, _) = self.char_indices().skip(char_idx).next().unwrap();
        self.remove(byte_idx);
    }
}

pub fn high_precision_sleep(duration: Duration, single_sleep_period: Duration) {
    let end_t = Instant::now() + duration;

    while Instant::now() < end_t {
        thread::sleep(single_sleep_period);
    }
}

pub unsafe fn slice_as_bytes<T>(slice: &[T]) -> &[u8] {
    slice::from_raw_parts(slice.as_ptr() as *const u8, mem::size_of::<T>() * slice.len())
}

pub fn calc_triangle_area(v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Vec3 {
    let side0 = v1 - v0;
    let side1 = v2 - v0;
    side0.cross(&side1) / 2.0
}

pub fn calc_triangle_normal(v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Vec3 {
    let side0 = v1 - v0;
    let side1 = v2 - v0;
    side0.cross(&side1).normalize()
}

pub fn smoothstep<S: RealNumber>(x: S) -> S {
    x * x * (glm::convert::<_, S>(3.0) - glm::convert::<_, S>(2.0) * x)
}

pub fn smootherstep<S: RealNumber>(x: S) -> S {
    x * x * x * (x * (x * glm::convert(6.0) - glm::convert(15.0)) + glm::convert(10.0))
}
