use crate::render_engine::RenderEngine;
use smallvec::SmallVec;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;

pub struct BufferResource {
    pub(in crate::render_engine) buffer: Vec<u8>,
    pub(in crate::render_engine) device_buffer: vkw::DeviceBuffer,
    pub(in crate::render_engine) changed: bool,
}

impl BufferResource {
    pub fn set(&mut self, device: &Arc<vkw::Device>, buffer: Vec<u8>) -> Result<(), vkw::DeviceError> {
        self.buffer = buffer;

        let curr_size = self.device_buffer.size();
        let new_size = self.buffer.len() as u64;

        if (new_size > curr_size) || (new_size < curr_size / 2) {
            self.device_buffer = device.create_device_buffer(
                vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
                new_size,
                1,
            )?;
        }

        self.changed = true;
        Ok(())
    }
}

pub enum Resource {
    Buffer(BufferResource),
}

impl Resource {
    pub fn buffer<T>(device: &Arc<vkw::Device>, buffer: &[T]) -> Result<Resource, vkw::DeviceError> {
        let size = buffer.len() * mem::size_of::<T>();

        let device_buffer = device.create_device_buffer(
            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
            size as u64,
            1,
        )?;
        let byte_slice = unsafe { slice::from_raw_parts(buffer.as_ptr() as *const u8, size) };

        Ok(Resource::Buffer(BufferResource {
            buffer: byte_slice.to_vec(),
            device_buffer,
            changed: true,
        }))
    }
}

// TODO GRAPHICS ENGINE OPTIMIZE: REMOVE USE OF `Arc`s

pub struct Renderer {
    pub(in crate::render_engine) mat_pipeline: u32,
    pub(in crate::render_engine) uniform_buffer_offset_model: u32,
    /// binding id -> Resource
    pub(in crate::render_engine) resources: SmallVec<[(u32, Resource); 4]>,
    pub(in crate::render_engine) translucent: bool,
    pub(in crate::render_engine) visible: bool,
}

impl Renderer {
    pub fn new(renderer: &RenderEngine, mat_pipeline: u32, translucent: bool) -> Renderer {
        let pipe = &renderer.material_pipelines[mat_pipeline as usize];

        Renderer {
            mat_pipeline,
            uniform_buffer_offset_model: pipe.uniform_buffer_offset_model(),
            resources: Default::default(),
            translucent,
            visible: true,
        }
    }

    pub fn visible(&self) -> bool {
        self.visible
    }

    pub fn set_visible(&mut self, visible: bool) {
        self.visible = visible;
    }

    pub fn resources_mut(&mut self) -> &mut SmallVec<[(u32, Resource); 4]> {
        &mut self.resources
    }
}
