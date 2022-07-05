use crate::{Device, PipelineSignature, RenderPass};
use ash::vk;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineStageFlags(pub(crate) vk::PipelineStageFlags);
vk_bitflags_impl!(PipelineStageFlags, vk::PipelineStageFlags);

impl PipelineStageFlags {
    pub const TOP_OF_PIPE: Self = Self(vk::PipelineStageFlags::TOP_OF_PIPE);
    pub const TRANSFER: Self = Self(vk::PipelineStageFlags::TRANSFER);
    pub const COLOR_ATTACHMENT_OUTPUT: Self = Self(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT);
    pub const VERTEX_SHADER: Self = Self(vk::PipelineStageFlags::VERTEX_SHADER);
    pub const PIXEL_SHADER: Self = Self(vk::PipelineStageFlags::FRAGMENT_SHADER);
    pub const LATE_FRAGMENT_TESTS: Self = Self(vk::PipelineStageFlags::LATE_FRAGMENT_TESTS);
    pub const COMPUTE: Self = Self(vk::PipelineStageFlags::COMPUTE_SHADER);
    pub const BOTTOM_OF_PIPE: Self = Self(vk::PipelineStageFlags::BOTTOM_OF_PIPE);
    pub const HOST: Self = Self(vk::PipelineStageFlags::HOST);
    pub const ALL_GRAPHICS: Self = Self(vk::PipelineStageFlags::ALL_GRAPHICS);
    pub const ALL_COMMANDS: Self = Self(vk::PipelineStageFlags::ALL_COMMANDS);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AccessFlags(pub(crate) vk::AccessFlags);
vk_bitflags_impl!(AccessFlags, vk::AccessFlags);

impl AccessFlags {
    pub const COLOR_ATTACHMENT_READ: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_READ);
    pub const COLOR_ATTACHMENT_WRITE: Self = Self(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);
    pub const DEPTH_ATTACHMENT_WRITE: Self = Self(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
    pub const UNIFORM_READ: Self = Self(vk::AccessFlags::UNIFORM_READ);
    pub const TRANSFER_READ: Self = Self(vk::AccessFlags::TRANSFER_READ);
    pub const TRANSFER_WRITE: Self = Self(vk::AccessFlags::TRANSFER_WRITE);
    pub const SHADER_READ: Self = Self(vk::AccessFlags::SHADER_READ);
    pub const SHADER_WRITE: Self = Self(vk::AccessFlags::SHADER_WRITE);
    pub const MEMORY_READ: Self = Self(vk::AccessFlags::MEMORY_READ);
    pub const MEMORY_WRITE: Self = Self(vk::AccessFlags::MEMORY_WRITE);
    pub const HOST_READ: Self = Self(vk::AccessFlags::HOST_READ);
    pub const HOST_WRITE: Self = Self(vk::AccessFlags::HOST_WRITE);
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PrimitiveTopology(pub(crate) vk::PrimitiveTopology);

impl PrimitiveTopology {
    pub const TRIANGLE_LIST: Self = Self(vk::PrimitiveTopology::TRIANGLE_LIST);
    pub const TRIANGLE_STRIP: Self = Self(vk::PrimitiveTopology::TRIANGLE_STRIP);
}

pub struct PipelineDepthStencil {
    pub depth_test: bool,
    pub depth_write: bool,
    pub stencil_test: bool,
}

impl PipelineDepthStencil {
    pub fn new() -> PipelineDepthStencil {
        Default::default()
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

impl Default for PipelineDepthStencil {
    fn default() -> Self {
        PipelineDepthStencil {
            depth_test: false,
            depth_write: false,
            stencil_test: false,
        }
    }
}

pub struct PipelineRasterization {
    pub cull_back_faces: bool,
}

impl PipelineRasterization {
    pub fn new() -> PipelineRasterization {
        Default::default()
    }

    pub fn cull_back_faces(mut self, enabled: bool) -> PipelineRasterization {
        self.cull_back_faces = enabled;
        self
    }
}

impl Default for PipelineRasterization {
    fn default() -> Self {
        PipelineRasterization {
            cull_back_faces: false,
        }
    }
}

#[derive(Copy, Clone)]
pub struct AttachmentColorBlend(pub(crate) vk::PipelineColorBlendAttachmentState);

impl AttachmentColorBlend {
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.0.blend_enable = enabled as u32;
        self
    }
}

impl Default for AttachmentColorBlend {
    fn default() -> Self {
        Self(
            vk::PipelineColorBlendAttachmentState::builder()
                .blend_enable(false)
                .color_blend_op(vk::BlendOp::ADD)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_write_mask(
                    vk::ColorComponentFlags::R
                        | vk::ColorComponentFlags::G
                        | vk::ColorComponentFlags::B
                        | vk::ColorComponentFlags::A,
                )
                .build(),
        )
    }
}

pub struct Pipeline {
    pub(crate) device: Arc<Device>,
    pub(crate) _render_pass: Option<Arc<RenderPass>>,
    pub(crate) signature: Arc<PipelineSignature>,
    pub(crate) native: vk::Pipeline,
    pub(crate) bind_point: vk::PipelineBindPoint,
}

impl Pipeline {
    pub fn signature(&self) -> &Arc<PipelineSignature> {
        &self.signature
    }
}

impl Eq for Pipeline {}

impl PartialEq for Pipeline {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Hash for Pipeline {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.native.hash(state);
    }
}

impl Drop for Pipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.wrapper.native.destroy_pipeline(self.native, None);
        }
    }
}
