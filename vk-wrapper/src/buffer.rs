use crate::Device;
use ash::vk;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use std::{marker::PhantomData, mem, ptr, rc::Rc};

pub(crate) struct Buffer {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::Buffer,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) aligned_elem_size: u64,
    pub(crate) size: u64,
    pub(crate) _bytesize: u64,
}

#[derive(Clone)]
pub struct RawHostBuffer(Arc<Buffer>);

unsafe impl Send for RawHostBuffer {}
unsafe impl Sync for RawHostBuffer {}

#[derive(Clone)]
pub struct BufferBarrier(pub(crate) vk::BufferMemoryBarrier);

pub struct HostBuffer<T> {
    pub(crate) _type_marker: PhantomData<T>,
    pub(crate) buffer: Arc<Buffer>,
    pub(crate) p_data: *mut u8,
}

impl<'a, T> IntoIterator for &'a HostBuffer<T> {
    type Item = &'a mut T;
    type IntoIter = HostBufferIterator<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        HostBufferIterator {
            p_data: self.p_data,
            stride: self.buffer.aligned_elem_size as usize,
            size: self.buffer.size as usize,
            _marker: PhantomData,
            index: 0,
        }
    }
}

pub struct HostBufferIterator<'a, T> {
    p_data: *mut u8,
    stride: usize,
    size: usize,
    index: usize,
    _marker: PhantomData<&'a T>,
}

impl<'a, T> Iterator for HostBufferIterator<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.size {
            let curr_index = self.index;
            self.index += 1;
            Some(unsafe { &mut *(self.p_data.offset((self.stride * curr_index) as isize) as *mut T) })
        } else {
            None
        }
    }
}

impl<T> Index<usize> for HostBuffer<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        if index as u64 >= self.buffer.size {
            panic!(
                "VkBuffer: index out of bounds: the len is {} but the index is {}",
                self.buffer.size, index
            );
        }

        unsafe {
            &*(self
                .p_data
                .offset(self.buffer.aligned_elem_size as isize * index as isize) as *const T)
        }
    }
}

impl<T> IndexMut<usize> for HostBuffer<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index as u64 >= self.buffer.size {
            panic!(
                "VkBuffer: index out of bounds: the len is {} but the index is {}",
                self.buffer.size, index
            );
        }
        unsafe {
            &mut *(self
                .p_data
                .offset(self.buffer.aligned_elem_size as isize * index as isize) as *mut T)
        }
    }
}

impl<T> HostBuffer<T> {
    pub fn get_raw(&self) -> RawHostBuffer {
        RawHostBuffer(Arc::clone(&self.buffer))
    }

    pub fn read(&self, first_element: u64, elements: &mut [T]) {
        if first_element + elements.len() as u64 >= self.buffer.size {
            panic!(
                "VkBuffer: index {} out of range for slice of length {}",
                first_element + elements.len() as u64,
                elements.len()
            );
        }

        if mem::size_of::<T>() == self.buffer.aligned_elem_size as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    (self.p_data as *const T).offset(first_element as isize),
                    elements.as_mut_ptr(),
                    elements.len(),
                )
            };
        } else {
            for i in 0..elements.len() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (self.p_data as *const u8).offset(
                            (first_element as isize + i as isize) * self.buffer.aligned_elem_size as isize,
                        ),
                        (elements.as_mut_ptr()).offset(first_element as isize + i as isize) as *mut u8,
                        1,
                    );
                }
            }
        }
    }

    pub fn write(&mut self, first_element: u64, elements: &[T]) {
        if first_element + elements.len() as u64 >= self.buffer.size {
            panic!(
                "VkBuffer: index {} out of range for slice of length {}",
                first_element + elements.len() as u64,
                elements.len()
            );
        }

        if mem::size_of::<T>() == self.buffer.aligned_elem_size as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    elements.as_ptr(),
                    (self.p_data as *mut T).offset(first_element as isize),
                    elements.len(),
                )
            };
        } else {
            for i in 0..elements.len() {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (elements.as_ptr()).offset(first_element as isize + i as isize) as *mut u8,
                        (self.p_data as *mut u8).offset(
                            (first_element as isize + i as isize) * self.buffer.aligned_elem_size as isize,
                        ),
                        1,
                    );
                }
            }
        }
    }
}

unsafe impl<T> Send for HostBuffer<T> {}
unsafe impl<T> Sync for HostBuffer<T> {}

impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        self.buffer
            .device
            .allocator
            .destroy_buffer(self.buffer.native, &self.buffer.allocation)
            .unwrap();
    }
}

pub struct DeviceBuffer {
    pub(crate) buffer: Buffer,
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        self.buffer
            .device
            .allocator
            .destroy_buffer(self.buffer.native, &self.buffer.allocation)
            .unwrap();
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferUsageFlags(pub(crate) vk::BufferUsageFlags);
vk_bitflags_impl!(BufferUsageFlags, vk::BufferUsageFlags);

impl BufferUsageFlags {
    pub const TRANSFER_SRC: Self = Self(vk::BufferUsageFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::BufferUsageFlags::TRANSFER_DST);
    pub const VERTEX: Self = Self(vk::BufferUsageFlags::VERTEX_BUFFER);
    pub const INDEX: Self = Self(vk::BufferUsageFlags::INDEX_BUFFER);
    pub const UNIFORM: Self = Self(vk::BufferUsageFlags::UNIFORM_BUFFER);
    pub const INDIRECT: Self = Self(vk::BufferUsageFlags::INDIRECT_BUFFER);
    pub const STORAGE: Self = Self(vk::BufferUsageFlags::STORAGE_BUFFER);
}