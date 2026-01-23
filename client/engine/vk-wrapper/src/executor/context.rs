use crate::{
    AttachmentColorBlend, BufferHandle, BufferUsageFlags, CmdList, DescriptorPool, Device, DeviceBuffer,
    DeviceError, Fence, Format, HostBuffer, Image, ImageLayout, ImageType, ImageUsageFlags, Pipeline,
    PipelineDepthStencil, PipelineOutputInfo, PipelineRasterization, PrimitiveTopology, Semaphore, Shader,
    buffer::BufferHandleImpl,
    image::ImageParams,
    pipeline::{CompareOp, CullMode},
};
use ash::vk::FormatFeatureFlags;
use common::{parking_lot::Mutex, types::HashMap};
use generational_arena::Arena;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CPUBufferId(generational_arena::Index);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GPUBufferId(generational_arena::Index);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GPUImageId(generational_arena::Index);

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct GPUPipelineId(generational_arena::Index);

#[derive(Clone, Copy)]
pub enum GPUResourceId {
    Buffer(GPUBufferId),
    Image(GPUImageId),
}

pub(crate) struct CPUBuffer {
    pub(crate) inner: HostBuffer<u8>,
}

pub(crate) struct GPUBuffer {
    pub(crate) inner: DeviceBuffer,
    pub(crate) name: Option<String>,
}

pub(crate) struct GPUImage {
    pub(crate) inner: Arc<Image>,
    pub(crate) name: Option<String>,
    pub(crate) last_layout: ImageLayout,
}

pub(crate) struct GPUPipeline {
    pub(crate) inner: Arc<Pipeline>,
    // pub(crate) bindings_mem_usages: HashMap<BindingLoc, MemAccess>,
    pub(crate) set0_descriptor_pool: DescriptorPool,
    pub(crate) set1_descriptor_pool: DescriptorPool,
    pub(crate) name: Option<String>,
}

pub struct GPUImageParams {
    pub(crate) ty: ImageType,
    pub(crate) format: Format,
    pub(crate) preferred_size: (u32, u32, u32),
    pub(crate) preferred_mip_levels: u32,
    pub(crate) is_array: bool,
}

impl GPUImageParams {
    pub fn d2(format: Format, preferred_size: (u32, u32)) -> Self {
        Self {
            ty: Image::TYPE_2D,
            format,
            preferred_size: (preferred_size.0, preferred_size.1, 1),
            preferred_mip_levels: 1,
            is_array: false,
        }
    }

    pub fn d2_array(format: Format, preferred_size: (u32, u32, u32)) -> Self {
        Self {
            ty: Image::TYPE_2D,
            format,
            preferred_size,
            preferred_mip_levels: 1,
            is_array: true,
        }
    }

    pub fn d3(format: Format, preferred_size: (u32, u32, u32)) -> Self {
        Self {
            ty: Image::TYPE_3D,
            format,
            preferred_size,
            preferred_mip_levels: 1,
            is_array: false,
        }
    }

    /// If max_mip_levels = 0, mip level count is calculated automatically.
    pub fn with_preferred_mip_levels(mut self, max_mip_levels: u32) -> Self {
        self.preferred_mip_levels = max_mip_levels;
        self
    }
}

pub struct DrawAttachmentParams {
    format: Format,
    blend: AttachmentColorBlend,
}

pub struct GPUPipelineParams {
    topology: PrimitiveTopology,
    shaders: Vec<Arc<Shader>>,
    depth_test: bool,
    depth_write: bool,
    depth_compare: CompareOp,
    cull: CullMode,
    attachments_configs: Vec<DrawAttachmentParams>,
}

pub enum CPUBufferInfo<'a> {
    Bytes(&'a [u8]),
    Size(u64),
}

impl CPUBufferInfo<'_> {
    fn size(&self) -> u64 {
        match self {
            CPUBufferInfo::Bytes(slice) => slice.len() as u64,
            CPUBufferInfo::Size(size) => *size,
        }
    }
}

pub struct GPUContext {
    pub(crate) device: Arc<Device>,
    cpu_buffers: Arena<CPUBuffer>,
    buffers: Arena<GPUBuffer>,
    buffers_name2id: HashMap<String, GPUBufferId>,
    images: Arena<GPUImage>,
    images_name2id: HashMap<String, GPUImageId>,
    pipelines: Arena<GPUPipeline>,
    pipelines_name2id: HashMap<String, GPUPipelineId>,
    pub(crate) temp_buffers: Vec<HostBuffer<u8>>,
    pub(crate) cmd_list: Arc<Mutex<CmdList>>,
    pub(crate) finish_semaphore: Arc<Semaphore>,
    pub(crate) finish_fence: Fence,
}

impl GPUContext {
    pub fn new(device: Arc<Device>) -> Result<Self, DeviceError> {
        let queue = device.get_queue(crate::QueueType::Graphics);
        let cmd_list = queue.create_primary_cmd_list("main")?;
        let finish_semaphore = device.create_binary_semaphore()?;
        let finish_fence = device.create_fence()?;

        Ok(Self {
            device,
            cpu_buffers: Default::default(),
            buffers: Default::default(),
            buffers_name2id: Default::default(),
            images: Default::default(),
            images_name2id: Default::default(),
            pipelines: Default::default(),
            pipelines_name2id: Default::default(),
            temp_buffers: vec![],
            cmd_list: Arc::new(Mutex::new(cmd_list)),
            finish_semaphore: Arc::new(finish_semaphore),
            finish_fence,
        })
    }

    pub fn create_cpu_buffer(&mut self, info: CPUBufferInfo) -> Result<CPUBufferId, DeviceError> {
        let inner = self.device.create_host_buffer::<u8>(
            BufferUsageFlags::TRANSFER_SRC | BufferUsageFlags::TRANSFER_DST,
            info.size(),
        )?;
        let buffer_id = self.cpu_buffers.insert(CPUBuffer { inner });
        Ok(CPUBufferId(buffer_id))
    }

    pub(crate) fn get_cpu_buffer(&self, id: CPUBufferId) -> Option<&CPUBuffer> {
        self.cpu_buffers.get(id.0)
    }

    pub fn create_buffer(&mut self, size: u64) -> Result<GPUBufferId, DeviceError> {
        let inner = self.device.create_device_buffer(
            BufferUsageFlags::VERTEX
                | BufferUsageFlags::INDEX
                | BufferUsageFlags::TRANSFER_SRC
                | BufferUsageFlags::TRANSFER_DST
                | BufferUsageFlags::STORAGE
                | BufferUsageFlags::UNIFORM,
            1,
            size,
        )?;
        let buffer_id = self.buffers.insert(GPUBuffer { inner, name: None });
        Ok(GPUBufferId(buffer_id))
    }

    pub fn create_buffer_named(&mut self, size: u64, name: &str) -> Result<(), DeviceError> {
        let buffer_id = self.create_buffer(size)?;
        self.buffers_name2id.insert(name.to_string(), buffer_id);
        Ok(())
    }

    pub fn destroy_buffer(&mut self, id: GPUBufferId) {
        let buffer = self.buffers.remove(id.0).expect("buffer id should be valid");
        if let Some(name) = &buffer.name {
            self.buffers_name2id.remove(name);
        }
    }

    pub(crate) fn get_buffer(&self, id: GPUBufferId) -> Option<&GPUBuffer> {
        self.buffers.get(id.0)
    }

    pub(crate) fn create_temp_host_buffer(
        &mut self,
        info: CPUBufferInfo,
    ) -> Result<BufferHandle, DeviceError> {
        let mut buffer = self
            .device
            .create_host_buffer(BufferUsageFlags::TRANSFER_SRC, info.size())?;
        let handle = buffer.handle();

        if let CPUBufferInfo::Bytes(data) = info {
            buffer.write(0, data);
        }

        self.temp_buffers.push(buffer);
        Ok(handle)
    }

    pub fn clear_temp_buffers(&mut self) {
        self.temp_buffers.clear();
    }

    pub fn create_image(&mut self, params: GPUImageParams) -> Result<GPUImageId, DeviceError> {
        let inner_params = ImageParams {
            ty: params.ty,
            format: params.format,
            usage: {
                let mut usage = ImageUsageFlags::empty();
                let features = self.device.adapter().formats_props[&params.format.0].optimal_tiling_features;

                if features.contains(FormatFeatureFlags::COLOR_ATTACHMENT) {
                    usage |= ImageUsageFlags::COLOR_ATTACHMENT;
                }
                if features.contains(FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
                    usage |= ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
                }
                if features.contains(FormatFeatureFlags::STORAGE_IMAGE) {
                    usage |= ImageUsageFlags::STORAGE;
                }
                if features.contains(FormatFeatureFlags::SAMPLED_IMAGE) {
                    usage |= ImageUsageFlags::SAMPLED;
                }
                if features.contains(FormatFeatureFlags::TRANSFER_SRC) {
                    usage |= ImageUsageFlags::TRANSFER_SRC;
                }
                if features.contains(FormatFeatureFlags::TRANSFER_DST) {
                    usage |= ImageUsageFlags::TRANSFER_DST;
                }
                usage
            },
            preferred_size: params.preferred_size,
            preferred_mip_levels: params.preferred_mip_levels,
            is_array: params.is_array,
        };
        let inner = { self.device.create_image(&inner_params, "")? };
        let inner_id = self.images.insert(GPUImage {
            inner,
            name: None,
            last_layout: ImageLayout::UNDEFINED,
        });
        Ok(GPUImageId(inner_id))
    }

    pub fn destroy_image(&mut self, id: GPUImageId) {
        let image = self.images.remove(id.0).expect("image id should be valid");
        if let Some(name) = &image.name {
            self.images_name2id.remove(name);
        }
    }

    pub(crate) fn get_image(&self, id: GPUImageId) -> Option<&GPUImage> {
        self.images.get(id.0)
    }

    pub(crate) fn iter_images(&self) -> impl Iterator<Item = (GPUImageId, &GPUImage)> {
        self.images.iter().map(|(key, img)| (GPUImageId(key), img))
    }

    pub fn create_pipeline(&mut self, params: GPUPipelineParams) -> Result<GPUPipelineId, DeviceError> {
        let signature = self.device.create_pipeline_signature(&params.shaders, &[])?;

        // let bindings_mem_usages: HashMap<_, _> = signature
        //     .bindings
        //     .iter()
        //     .map(|(key, val)| (*key, MemAccess::from_flags(val.readable, val.writable)))
        //     .collect();

        let inner = self
            .device
            .create_graphics_pipeline(
                PipelineOutputInfo::DynamicRender(
                    params.attachments_configs.iter().map(|v| v.format).collect(),
                ),
                params.topology,
                PipelineDepthStencil::new()
                    .depth_test(params.depth_test)
                    .depth_write(params.depth_write)
                    .depth_compare_op(params.depth_compare),
                PipelineRasterization::new().cull(params.cull),
                &params
                    .attachments_configs
                    .iter()
                    .enumerate()
                    .map(|(idx, v)| (idx as u32, v.blend))
                    .collect::<Vec<_>>(),
                &signature,
                &[],
            )
            .unwrap();

        let set0_descriptor_pool = signature.create_pool(0, 16, "pool")?;
        let set1_descriptor_pool = signature.create_pool(1, 64, "pool")?;

        let inner_id = self.pipelines.insert(GPUPipeline {
            inner,
            // bindings_mem_usages,
            name: None,
            set0_descriptor_pool,
            set1_descriptor_pool,
        });
        Ok(GPUPipelineId(inner_id))
    }

    pub fn destroy_pipeline(&mut self, id: GPUPipelineId) {
        let pipeline = self.pipelines.remove(id.0).expect("pipeline id should be valid");
        if let Some(name) = &pipeline.name {
            self.pipelines_name2id.remove(name);
        }
    }

    pub(crate) fn get_pipeline(&self, id: GPUPipelineId) -> Option<&GPUPipeline> {
        self.pipelines.get(id.0)
    }

    pub(crate) fn get_pipeline_mut(&mut self, id: GPUPipelineId) -> Option<&mut GPUPipeline> {
        self.pipelines.get_mut(id.0)
    }
}
