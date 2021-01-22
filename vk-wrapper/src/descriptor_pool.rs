use crate::{Device, DeviceBuffer, Image, ImageLayout, PipelineSignature};
use ahash::AHashMap;
use ash::version::DeviceV1_0;
use ash::vk;
use bit_set::BitSet;
use smallvec::SmallVec;
use std::ops::Deref;
use std::slice;
use std::sync::{Arc, Mutex};

pub enum BindingRes {
    Buffer(Arc<DeviceBuffer>),
    /// [buffer, offset, range]
    BufferRange(Arc<DeviceBuffer>, u64, u64),
    Image(Arc<Image>, ImageLayout),
}

pub struct Binding {
    pub id: u32,
    pub array_index: u32,
    pub res: BindingRes,
}

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct BindingMapping {
    descriptor_set_id: u32,
    binding_id: u32,
    array_index: u32,
}

pub struct DescriptorPoolWrapper {
    pub(crate) device: Arc<Device>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) set_layout_id: u32,
    pub(crate) native: vk::DescriptorPool,
    pub(crate) allocated: Vec<vk::DescriptorSet>,
    pub(crate) free_sets: BitSet,
    pub(crate) _used_buffers: AHashMap<BindingMapping, Arc<DeviceBuffer>>,
    pub(crate) _used_images: AHashMap<BindingMapping, Arc<Image>>,
}

pub struct DescriptorPool(pub(in crate) Mutex<DescriptorPoolWrapper>);

impl DescriptorPoolWrapper {
    pub fn alloc(&mut self) -> Result<u32, vk::Result> {
        if let Some(id) = self.free_sets.iter().next() {
            self.free_sets.remove(id);
            Ok(id as u32)
        } else {
            let alloc_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(self.native)
                .set_layouts(slice::from_ref(
                    &self.signature.native[self.set_layout_id as usize],
                ));

            self.allocated
                .push(unsafe { self.device.wrapper.0.allocate_descriptor_sets(&alloc_info)?[0] });
            Ok((self.allocated.len() - 1) as u32)
        }
    }

    pub fn free(&mut self, descriptor_set_id: u32) {
        self.free_sets.insert(descriptor_set_id as usize);
    }

    pub fn update(&mut self, descriptor_set_id: u32, updates: &[Binding]) {
        if self.free_sets.contains(descriptor_set_id as usize) {
            panic!("descriptor set isn't allocated!");
        }

        let native_set = self.allocated[descriptor_set_id as usize];

        let mut native_buffer_infos = SmallVec::<[vk::DescriptorBufferInfo; 8]>::with_capacity(updates.len());
        let mut native_image_infos = SmallVec::<[vk::DescriptorImageInfo; 8]>::with_capacity(updates.len());
        let mut native_writes = SmallVec::<[vk::WriteDescriptorSet; 8]>::with_capacity(updates.len());

        for binding in updates {
            let mut write_info = vk::WriteDescriptorSet::builder()
                .dst_set(native_set)
                .dst_binding(binding.id)
                .dst_array_element(binding.array_index)
                .descriptor_type(self.signature.binding_types[&binding.id]);
            let mapping = BindingMapping {
                descriptor_set_id,
                binding_id: binding.id,
                array_index: binding.array_index,
            };

            match &binding.res {
                BindingRes::Buffer(buffer) => {
                    native_buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: buffer.buffer.native,
                        offset: 0,
                        range: vk::WHOLE_SIZE,
                    });
                    write_info = write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));

                    self._used_buffers.insert(mapping, Arc::clone(buffer));
                }
                BindingRes::BufferRange(buffer, offset, range) => {
                    native_buffer_infos.push(vk::DescriptorBufferInfo {
                        buffer: buffer.buffer.native,
                        offset: *offset,
                        range: *range,
                    });
                    write_info = write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));

                    self._used_buffers.insert(mapping, Arc::clone(buffer));
                }
                BindingRes::Image(image, layout) => {
                    native_image_infos.push(vk::DescriptorImageInfo {
                        sampler: image.sampler,
                        image_view: image.view,
                        image_layout: layout.0,
                    });
                    write_info = write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));

                    self._used_images.insert(mapping, Arc::clone(image));
                }
            }

            native_writes.push(write_info.build());
        }

        unsafe { self.device.wrapper.0.update_descriptor_sets(&native_writes, &[]) };
    }
}

impl Drop for DescriptorPoolWrapper {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_descriptor_pool(self.native, None) };
    }
}

impl Deref for DescriptorPool {
    type Target = Mutex<DescriptorPoolWrapper>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
