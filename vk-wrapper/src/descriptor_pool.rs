use crate::{Device, DeviceBuffer, Image, ImageLayout, ImageView, PipelineSignature, Sampler};
use ash::version::DeviceV1_0;
use ash::vk;
use bit_set::BitSet;
use smallvec::SmallVec;
use std::slice;
use std::sync::Arc;

pub enum BindingRes<'a> {
    Buffer(&'a DeviceBuffer),
    Image(Arc<Image>, ImageLayout),
    ImageView(Arc<ImageView>, ImageLayout),
    ImageViewSampler(Arc<ImageView>, Arc<Sampler>, ImageLayout),
}

pub struct Binding<'a> {
    pub id: u32,
    pub array_index: u32,
    pub res: BindingRes<'a>,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(crate) struct BindingMapping {
    descriptor_set_id: u32,
    binding_id: u32,
    array_index: u32,
}

#[derive(Copy, Clone)]
pub struct DescriptorSet {
    pub(crate) native: vk::DescriptorSet,
    pub(crate) id: u32,
}

pub struct NativeDescriptorPool {
    pub(crate) handle: vk::DescriptorPool,
    pub(crate) size: u32,
}

pub struct DescriptorPool {
    pub(crate) device: Arc<Device>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) set_layout_id: u32,
    pub(crate) native: SmallVec<[NativeDescriptorPool; 16]>,
    pub(crate) allocated: Vec<DescriptorSet>,
    pub(crate) free_sets: BitSet,
}

impl DescriptorPool {
    pub fn alloc(&mut self) -> Result<DescriptorSet, vk::Result> {
        if let Some(id) = self.free_sets.iter().next() {
            self.free_sets.remove(id);
            Ok(self.allocated[id])
        } else {
            let last_size = self.native.last().unwrap().size;
            let next_size = (last_size + 1).next_power_of_two();
            let next_pool = NativeDescriptorPool {
                handle: self.signature.create_native_pool(self.set_layout_id, next_size)?,
                size: next_size,
            };
            let layouts = vec![self.signature.native[self.set_layout_id as usize]; next_size as usize];
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(next_pool.handle)
                .set_layouts(&layouts);
            let sets = unsafe { self.device.wrapper.0.allocate_descriptor_sets(&alloc_info)? };

            let start_i = self.allocated.len() as u32;
            self.allocated
                .extend(sets.into_iter().enumerate().map(|(i, v)| DescriptorSet {
                    native: v,
                    id: start_i + i as u32,
                }));
            self.free_sets
                .extend((start_i as usize..(start_i + next_size) as usize).into_iter());
            self.native.push(next_pool);

            let id = self.free_sets.iter().next().unwrap();
            self.free_sets.remove(id);
            Ok(self.allocated[id])
        }
    }

    pub fn free(&mut self, descriptor_set: DescriptorSet) {
        self.free_sets.insert(descriptor_set.id as usize);
    }

    pub fn update(&mut self, descriptor_set: DescriptorSet, updates: &[Binding]) {
        if self.free_sets.contains(descriptor_set.id as usize) {
            panic!("descriptor set isn't allocated!");
        }
        if updates.is_empty() {
            return;
        }

        let mut native_buffer_infos = SmallVec::<[vk::DescriptorBufferInfo; 8]>::with_capacity(updates.len());
        let mut native_image_infos = SmallVec::<[vk::DescriptorImageInfo; 8]>::with_capacity(updates.len());
        let mut native_writes = SmallVec::<[vk::WriteDescriptorSet; 8]>::with_capacity(updates.len());

        for binding in updates {
            let mut write_info = vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set.native)
                .dst_binding(binding.id)
                .dst_array_element(binding.array_index)
                .descriptor_type(self.signature.binding_types[self.set_layout_id as usize][&binding.id]);

            match &binding.res {
                BindingRes::Buffer(buffer) => {
                    native_buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: buffer.buffer.native,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    });
                    write_info = write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));
                }
                BindingRes::Image(image, layout) => {
                    native_image_infos.push(vk::DescriptorImageInfo {
                        sampler: image.sampler.native,
                        image_view: image.view.native,
                        image_layout: layout.0,
                    });
                    write_info = write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                }
                BindingRes::ImageView(image_view, layout) => {
                    native_image_infos.push(vk::DescriptorImageInfo {
                        sampler: Default::default(),
                        image_view: image_view.native,
                        image_layout: layout.0,
                    });
                    write_info = write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                }
                BindingRes::ImageViewSampler(image_view, sampler, layout) => {
                    native_image_infos.push(vk::DescriptorImageInfo {
                        sampler: sampler.native,
                        image_view: image_view.native,
                        image_layout: layout.0,
                    });
                    write_info = write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                }
            }

            native_writes.push(write_info.build());
        }

        unsafe { self.device.wrapper.0.update_descriptor_sets(&native_writes, &[]) };
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        for native in &self.native {
            unsafe { self.device.wrapper.0.destroy_descriptor_pool(native.handle, None) };
        }
    }
}
