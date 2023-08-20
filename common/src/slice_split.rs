use std::{
    cmp::Ordering,
    ops::{Index, IndexMut},
};

pub struct SliceSplit<'a, T> {
    left: &'a mut [T],
    mid_index: usize,
    right: &'a mut [T],
}

impl<T> SliceSplit<'_, T> {
    pub fn get(&self, index: usize) -> Option<&T> {
        match index.cmp(&self.mid_index) {
            Ordering::Less => self.left.get(index),
            Ordering::Greater => self.right.get(index - self.left.len() - 1),
            _ => None,
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        match index.cmp(&self.mid_index) {
            Ordering::Less => self.left.get_mut(index),
            Ordering::Greater => self.right.get_mut(index - self.left.len() - 1),
            _ => None,
        }
    }
}

impl<T> Index<usize> for SliceSplit<'_, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).unwrap()
    }
}

impl<T> IndexMut<usize> for SliceSplit<'_, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index).unwrap()
    }
}

pub trait SliceSplitImpl<T> {
    fn split_mid_mut(&mut self, index: usize) -> Option<(&mut T, SliceSplit<T>)>;
}

impl<T> SliceSplitImpl<T> for [T] {
    fn split_mid_mut(&mut self, index: usize) -> Option<(&mut T, SliceSplit<T>)> {
        let (left, right) = self.split_at_mut(index);

        right.split_first_mut().map(move |(mid, right)| {
            (
                mid,
                SliceSplit {
                    left,
                    mid_index: index,
                    right,
                },
            )
        })
    }
}
