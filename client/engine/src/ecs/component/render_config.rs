use smallvec::SmallVec;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;

pub struct BufferResource {
    pub(crate) buffer: Vec<u8>,
    pub(crate) device_buffer: vkw::DeviceBuffer,
    pub(crate) changed: bool,
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

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum RenderType {
    /// Renders regular 3D objects.
    Main = 0,
    /// Renders objects after the `MAIN` stage.
    Overlay = 1,
    MainShadowMap = 2,
}

pub struct MeshRenderConfigC {
    pub(crate) render_ty: RenderType,
    pub(crate) mat_pipeline: u32,
    /// binding id -> Resource
    pub(crate) resources: SmallVec<[(u32, Resource); 4]>,
    pub(crate) translucent: bool,
    pub(crate) visible: bool,
}

impl Default for MeshRenderConfigC {
    fn default() -> Self {
        Self {
            render_ty: RenderType::Main,
            mat_pipeline: u32::MAX,
            resources: Default::default(),
            translucent: false,
            visible: false,
        }
    }
}

impl MeshRenderConfigC {
    pub fn new(mat_pipeline: u32, translucent: bool) -> MeshRenderConfigC {
        MeshRenderConfigC {
            render_ty: RenderType::Main,
            mat_pipeline,
            resources: Default::default(),
            translucent,
            visible: true,
        }
    }

    pub fn with_stage(mut self, stage: RenderType) -> Self {
        self.render_ty = stage;
        self
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
