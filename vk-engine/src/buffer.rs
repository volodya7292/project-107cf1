use crate::device::Device;
use ash::vk;
use std::ops::{Index, IndexMut, Range};
use std::{marker::PhantomData, mem, ptr, rc::Rc};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BufferUsageFlags(pub(crate) vk::BufferUsageFlags);
vk_bitflags_impl!(BufferUsageFlags, vk::BufferUsageFlags);

pub(crate) struct Buffer<T: ?Sized> {
    pub(crate) _device: Rc<Device>,
    pub(crate) _type_marker: PhantomData<T>,
    pub(crate) native: vk::Buffer,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) aligned_elem_size: u64,
    pub(crate) size: u64,
    pub(crate) bytesize: u64,
}

pub struct HostBuffer<T> {
    pub(crate) buffer: Buffer<T>,
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

// TODO: CONSIDER ALIGNMENT

// impl<T> Index<Range<usize>> for HostBuffer<T> {
//     type Output = [T];

//     fn index(&self, range: Range<usize>) -> &Self::Output {
//         if range.end as u64 >= self.buffer.size {
//             panic!("VkBuffer: index {} out of range for slice of length {}", range.end, range.end - range.start);
//         }
//         unsafe {
//             std::slice::from_raw_parts(
//                 self.p_data
//                     .offset(self.buffer.aligned_elem_size as isize * range.start as isize)
//                     as *const T,
//                 range.end - range.start,
//             )
//         }
//     }
// }

// impl<T> IndexMut<Range<usize>> for HostBuffer<T> {
//     fn index_mut(&mut self, range: Range<usize>) -> &mut Self::Output {
//         if range.end as u64 >= self.buffer.size {
//             panic!("VkBuffer: index {} out of range for slice of length {}", range.end, range.end - range.start);
//         }
//         unsafe {
//             std::slice::from_raw_parts_mut(
//                 self.p_data
//                     .offset(self.buffer.aligned_elem_size as isize * range.start as isize)
//                     as *mut T,
//                 range.end - range.start,
//             )
//         }
//     }
// }

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
    fn read(&self, first_element: u64, elements: &mut [T]) {
        let read_count = (elements.len() as isize)
            .min(
                self.buffer.bytesize as isize / self.buffer.aligned_elem_size as isize
                    - first_element as isize,
            )
            .max(0);

        if read_count == 0 {
            return;
        }

        if mem::size_of::<T>() == self.buffer.aligned_elem_size as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    (self.p_data as *const T).offset(first_element as isize),
                    elements.as_mut_ptr(),
                    read_count as usize,
                )
            };
        } else {
            for i in 0..read_count {
                unsafe {
                    ptr::copy_nonoverlapping(
                        (self.p_data as *const u8)
                            .offset((first_element as isize + i) * self.buffer.aligned_elem_size as isize),
                        (elements.as_mut_ptr()).offset(first_element as isize + i) as *mut u8,
                        1,
                    );
                }
            }
        }
    }

    fn write(&self, first_element: u64, elements: &[T]) {
        // TODO: range check
        // TODO: consider element alignment

        unsafe {
            ptr::copy_nonoverlapping(
                elements.as_ptr(),
                (self.p_data as *mut T).offset(first_element as isize),
                elements.len(),
            )
        };
    }
}

pub struct DeviceBuffer<T> {
    pub(crate) buffer: Buffer<T>,
}

impl BufferUsageFlags {
    pub const TRANSFER_SRC: Self = Self(vk::BufferUsageFlags::TRANSFER_SRC);
    pub const TRANSFER_DST: Self = Self(vk::BufferUsageFlags::TRANSFER_DST);
    pub const VERTEX: Self = Self(vk::BufferUsageFlags::VERTEX_BUFFER);
    pub const INDEX: Self = Self(vk::BufferUsageFlags::INDEX_BUFFER);
    pub const UNIFORM: Self = Self(vk::BufferUsageFlags::UNIFORM_BUFFER);
    pub const INDIRECT: Self = Self(vk::BufferUsageFlags::INDIRECT_BUFFER);
    pub const STORAGE: Self = Self(vk::BufferUsageFlags::STORAGE_BUFFER);
}
