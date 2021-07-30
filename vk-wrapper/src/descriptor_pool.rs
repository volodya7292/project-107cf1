use crate::{BufferHandle, Device, Image, ImageLayout, ImageView, PipelineSignature, Sampler};
use ash::version::DeviceV1_0;
use ash::vk;
use bit_set::BitSet;
use smallvec::SmallVec;
use std::ops::Range;
use std::slice;
use std::sync::Arc;

pub enum BindingRes {
    Buffer(BufferHandle),
    Image(Arc<Image>, ImageLayout),
    ImageView(Arc<ImageView>, ImageLayout),
    ImageViewSampler(Arc<ImageView>, Arc<Sampler>, ImageLayout),
}

pub struct Binding {
    id: u32,
    ty: vk::DescriptorType,
    pub array_index: u32,
    pub res: BindingRes,
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
    pub(crate) fn alloc_next_pool(&mut self, size: u32) -> Result<(), vk::Result> {
        let next_pool = NativeDescriptorPool {
            handle: self.signature.create_native_pool(self.set_layout_id, size)?,
            size,
        };
        let layouts = vec![self.signature.native[self.set_layout_id as usize]; size as usize];
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
            .extend((start_i as usize..(start_i + size) as usize).into_iter());
        self.native.push(next_pool);
        Ok(())
    }

    pub fn alloc(&mut self) -> Result<DescriptorSet, vk::Result> {
        if let Some(id) = self.free_sets.iter().next() {
            self.free_sets.remove(id);
            Ok(self.allocated[id])
        } else {
            self.alloc_next_pool((self.native.last().unwrap().size + 1).next_power_of_two())?;
            self.alloc()
        }
    }

    pub fn free(&mut self, descriptor_set: DescriptorSet) {
        self.free_sets.insert(descriptor_set.id as usize);
    }

    pub fn create_binding(&self, id: u32, array_index: u32, res: BindingRes) -> Binding {
        let ty = self.signature.binding_types[self.set_layout_id as usize][&id];
        Binding {
            id,
            array_index,
            ty,
            res,
        }
    }
}

impl Drop for DescriptorPool {
    fn drop(&mut self) {
        for native in &self.native {
            unsafe { self.device.wrapper.0.destroy_descriptor_pool(native.handle, None) };
        }
    }
}

impl Device {
    pub unsafe fn update_descriptor_sets(
        &self,
        bindings: &[Binding],
        updates: &[(DescriptorSet, Range<usize>)],
    ) {
        if updates.is_empty() {
            return;
        }

        let mut native_buffer_infos = Vec::with_capacity(updates.len());
        let mut native_image_infos = Vec::with_capacity(updates.len());
        let mut native_writes = Vec::with_capacity(updates.len());

        for (descriptor_set, range) in updates {
            for i in range.clone() {
                let binding = &bindings[i];

                let mut write_info = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set.native)
                    .dst_binding(binding.id)
                    .dst_array_element(binding.array_index)
                    .descriptor_type(binding.ty);

                match &binding.res {
                    BindingRes::Buffer(handle) => {
                        native_buffer_infos.push(vk::DescriptorBufferInfo {
                            buffer: handle.0,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        });
                        write_info =
                            write_info.buffer_info(slice::from_ref(native_buffer_infos.last().unwrap()));
                    }
                    BindingRes::Image(image, layout) => {
                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler: image.sampler.native,
                            image_view: image.view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                    BindingRes::ImageView(image_view, layout) => {
                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler: Default::default(),
                            image_view: image_view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                    BindingRes::ImageViewSampler(image_view, sampler, layout) => {
                        native_image_infos.push(vk::DescriptorImageInfo {
                            sampler: sampler.native,
                            image_view: image_view.native,
                            image_layout: layout.0,
                        });
                        write_info =
                            write_info.image_info(slice::from_ref(native_image_infos.last().unwrap()));
                    }
                }

                native_writes.push(write_info.build());
            }
        }

        self.wrapper.0.update_descriptor_sets(&native_writes, &[]);
    }

    pub unsafe fn update_descriptor_set(&self, set: DescriptorSet, bindings: &[Binding]) {
        self.update_descriptor_sets(bindings, &[(set, 0..bindings.len())])
    }
}
