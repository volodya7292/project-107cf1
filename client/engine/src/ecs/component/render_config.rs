use crate::module::main_renderer;
use common::parking_lot::Mutex;
use common::resource_cache::CachedResource;
use smallvec::SmallVec;
use std::any::Any;
use std::sync::Arc;
use std::{mem, slice};
use vk_wrapper as vkw;
use vk_wrapper::image::ImageParams;
use vk_wrapper::shader::BindingId;
use vkw::buffer::BufferHandleImpl;

pub struct GPUBufferResource {
    pub(crate) new_source_data: Mutex<Option<Vec<u8>>>,
    pub(crate) name: String,
    pub(crate) buffer: Mutex<Option<vkw::DeviceBuffer>>,
}

impl GPUBufferResource {
    pub fn new<T: Copy>(name: impl Into<String>, source_data: &[T]) -> Arc<Self> {
        let size = std::mem::size_of_val(source_data);
        let byte_slice = unsafe { slice::from_raw_parts(source_data.as_ptr() as *const u8, size) };
        assert!(size > 0);

        Arc::new(GPUBufferResource {
            new_source_data: Mutex::new(Some(byte_slice.to_vec())),
            buffer: Mutex::new(None),
            name: name.into(),
        })
    }

    pub(crate) fn acquire_buffer(
        &self,
        device: &Arc<vkw::Device>,
    ) -> Result<vkw::BufferHandle, vkw::DeviceError> {
        let mut curr_buffer = self.buffer.lock();
        if let Some(buffer) = &*curr_buffer {
            return Ok(buffer.handle());
        }

        let size = self.new_source_data.lock().as_ref().unwrap().len() as u64;
        let buffer = device.create_device_buffer_named(
            vkw::BufferUsageFlags::TRANSFER_DST | vkw::BufferUsageFlags::STORAGE,
            size,
            1,
            &self.name,
        )?;
        let buffer_handle = buffer.handle();
        *curr_buffer = Some(buffer);

        Ok(buffer_handle)
    }
}

pub struct GPUImageResource {
    pub(crate) new_source_data: Mutex<Option<Vec<u8>>>,
    pub(crate) name: String,
    pub(crate) params: ImageParams,
    pub(crate) image: Mutex<Option<Arc<vkw::Image>>>,
}

impl GPUImageResource {
    pub fn new(name: impl Into<String>, params: ImageParams, source_data: Vec<u8>) -> Arc<Self> {
        assert!(!source_data.is_empty());
        let byte_slice = unsafe { slice::from_raw_parts(source_data.as_ptr(), source_data.len()) };

        Arc::new(GPUImageResource {
            new_source_data: Mutex::new(Some(byte_slice.to_vec())),
            params,
            image: Mutex::new(None),
            name: name.into(),
        })
    }

    pub fn size(&self) -> (u32, u32) {
        self.params.size_2d()
    }

    pub(crate) fn acquire_image(
        &self,
        device: &Arc<vkw::Device>,
    ) -> Result<Arc<vkw::Image>, vkw::DeviceError> {
        let mut curr_image = self.image.lock();
        if let Some(image) = &*curr_image {
            return Ok(Arc::clone(image));
        }

        let image = device.create_image(
            &self.params.add_usage(vkw::ImageUsageFlags::TRANSFER_DST),
            &self.name,
        )?;
        *curr_image = Some(Arc::clone(&image));

        Ok(image)
    }
}

impl CachedResource for GPUImageResource {
    fn footprint(&self) -> usize {
        if let Some(image) = &*self.image.lock() {
            return image.bytesize() as usize;
        }
        if let Some(source) = &*self.new_source_data.lock() {
            return source.len();
        }
        unreachable!()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Clone)]
pub enum GPUResource {
    Buffer(Arc<GPUBufferResource>),
    Image(Arc<GPUImageResource>),
    None,
}

impl PartialEq for GPUResource {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Buffer(b0), Self::Buffer(b1)) => Arc::ptr_eq(b0, b1),
            (Self::Image(i0), Self::Image(i1)) => Arc::ptr_eq(i0, i1),
            _ => mem::discriminant(self) == mem::discriminant(other),
        }
    }
}

impl From<Arc<GPUBufferResource>> for GPUResource {
    fn from(value: Arc<GPUBufferResource>) -> Self {
        Self::Buffer(value)
    }
}

impl From<Arc<GPUImageResource>> for GPUResource {
    fn from(value: Arc<GPUImageResource>) -> Self {
        Self::Image(value)
    }
}

impl Default for GPUResource {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RenderLayer {
    /// Renders regular 3D objects.
    Main,
    /// Renders objects after the `MAIN` stage.
    Overlay,
}

pub struct MeshRenderConfigC {
    pub(crate) render_layer: RenderLayer,
    pub(crate) mat_pipeline: u32,
    pub(crate) resources: SmallVec<[(BindingId, GPUResource); 4]>,
    pub(crate) translucent: bool,
}

impl Default for MeshRenderConfigC {
    fn default() -> Self {
        Self {
            render_layer: RenderLayer::Main,
            mat_pipeline: u32::MAX,
            resources: Default::default(),
            translucent: false,
        }
    }
}

impl MeshRenderConfigC {
    pub fn new(mat_pipeline: u32, translucent: bool) -> MeshRenderConfigC {
        MeshRenderConfigC {
            render_layer: RenderLayer::Main,
            mat_pipeline,
            resources: Default::default(),
            translucent,
        }
    }

    pub fn with_render_layer(mut self, layer: RenderLayer) -> Self {
        self.render_layer = layer;
        self
    }

    pub fn with_shader_resources(mut self, resources: SmallVec<[GPUResource; 4]>) -> Self {
        self.resources = resources
            .into_iter()
            .enumerate()
            .map(|(idx, res)| {
                let binding_id = main_renderer::CUSTOM_OBJ_BINDING_START_ID + idx as u32;
                (binding_id, res)
            })
            .collect();
        self
    }

    pub fn set_shader_resource(&mut self, idx: usize, res: GPUResource) {
        self.resources[idx].1 = res;
    }
}
