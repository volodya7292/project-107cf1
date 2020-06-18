use crate::Device;
use ash::version::DeviceV1_0;
use ash::vk;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineStageFlags(pub(crate) vk::PipelineStageFlags);
vk_bitflags_impl!(PipelineStageFlags, vk::PipelineStageFlags);

impl PipelineStageFlags {
    pub const TOP_OF_PIPE: Self = Self(vk::PipelineStageFlags::TOP_OF_PIPE);
    pub const TRANSFER: Self = Self(vk::PipelineStageFlags::TRANSFER);
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
    pub const PIXEL_SHADER: Self = Self(vk::PipelineStageFlags::FRAGMENT_SHADER);
    pub const BOTTOM_OF_PIPE: Self = Self(vk::PipelineStageFlags::BOTTOM_OF_PIPE);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AccessFlags(pub(crate) vk::AccessFlags);
vk_bitflags_impl!(AccessFlags, vk::AccessFlags);

impl AccessFlags {
    pub const COLOR_ATTACHMENT_READ: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_READ);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    pub const TRANSFER_READ: Self = Self(vk::AccessFlags::TRANSFER_READ);
    pub const TRANSFER_WRITE: Self = Self(vk::AccessFlags::TRANSFER_WRITE);
    pub const SHADER_READ: Self = Self(vk::AccessFlags::SHADER_READ);
    pub const SHADER_WRITE: Self = Self(vk::AccessFlags::SHADER_WRITE);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveTopology(pub(crate) vk::PrimitiveTopology);

pub struct PipelineDepthStencil {
    pub depth_test: bool,
    pub depth_write: bool,
    pub stencil_test: bool,
}

impl PipelineDepthStencil {
    pub fn new() -> PipelineDepthStencil {
        PipelineDepthStencil {
            depth_test: false,
            depth_write: false,
            stencil_test: false,
        }
    }

    pub fn depth_test(mut self, enabled: bool) -> PipelineDepthStencil {
        self.depth_test = enabled;
        self
    }

    pub fn depth_write(mut self, enabled: bool) -> PipelineDepthStencil {
        self.depth_write = enabled;
        self
    }

    pub fn stencil_test(mut self, enabled: bool) -> PipelineDepthStencil {
        self.stencil_test = enabled;
        self
    }
}

pub struct PipelineRasterization {
    pub cull_back_faces: bool,
}

impl PipelineRasterization {
    pub fn new() -> PipelineRasterization {
        PipelineRasterization {
            cull_back_faces: false,
        }
    }

    pub fn cull_back_faces(mut self, enabled: bool) -> PipelineRasterization {
        self.cull_back_faces = enabled;
        self
    }
}

pub struct PipelineColorBlend {}

impl PipelineColorBlend {
    pub fn new() -> PipelineColorBlend {
        PipelineColorBlend {}
    }
}

pub struct Pipeline {
    pub(crate) device: Arc<Device>,
    pub(crate) layout: vk::PipelineLayout,
    pub(crate) native: vk::Pipeline,
}

impl Pipeline {
    pub const TOPOLOGY_TRIANGLE_LIST: PrimitiveTopology =
        PrimitiveTopology(vk::PrimitiveTopology::TRIANGLE_LIST);
    pub const TOPOLOGY_TRIANGLE_STRIP: PrimitiveTopology =
        PrimitiveTopology(vk::PrimitiveTopology::TRIANGLE_STRIP);
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.wrapper.0.destroy_pipeline(self.native, None);
            self.device.wrapper.0.destroy_pipeline_layout(self.layout, None);
        }
    }
}