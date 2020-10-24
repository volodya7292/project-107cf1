use crate::format::DEPTH_FORMAT;
use crate::{
    device::DeviceError, AccessFlags, Device, Format, Framebuffer, Image, ImageLayout, ImageUsageFlags,
    PipelineStageFlags,
};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Copy, Clone)]
pub enum LoadStore {
    None,
    InitSave,
    InitClear,
    FinalSave,
    InitClearFinalSave,
    InitSaveFinalSave,
}

#[derive(Copy, Clone)]
pub struct Attachment {
    pub format: Format,
    pub init_layout: ImageLayout,
    pub final_layout: ImageLayout,
    pub load_store: LoadStore,
}

#[derive(Copy, Clone)]
pub struct AttachmentRef {
    pub index: u32,
    pub layout: ImageLayout,
}

#[derive(Clone)]
pub struct Subpass {
    pub color: Vec<AttachmentRef>,
    pub depth: Option<AttachmentRef>,
}

impl Subpass {
    pub const EXTERNAL: u32 = vk::SUBPASS_EXTERNAL;
}

pub struct SubpassDependency {
    pub src_subpass: u32,
    pub dst_subpass: u32,
    pub src_stage_mask: PipelineStageFlags,
    pub dst_stage_mask: PipelineStageFlags,
    pub src_access_mask: AccessFlags,
    pub dst_access_mask: AccessFlags,
}

pub struct RenderPass {
    pub(crate) device: Arc<Device>,
    pub(crate) native: vk::RenderPass,
    pub(crate) subpasses: Vec<Subpass>,
    pub(crate) attachments: Vec<Attachment>,
    pub(crate) _color_attachments: Vec<u32>,
    pub(crate) _depth_attachments: Vec<u32>,
}

#[derive(Clone)]
pub enum ImageMod {
    OverrideImage(Arc<Image>),
    AdditionalUsage(ImageUsageFlags),
}

pub enum ClearValue {
    Undefined,
    ColorF32([f32; 4]),
    ColorU32([u32; 4]),
    Depth(f32),
}

pub(crate) fn vk_clear_value(clear_value: &ClearValue) -> vk::ClearValue {
    match clear_value {
        ClearValue::Undefined => vk::ClearValue::default(),
        ClearValue::ColorF32(c) => vk::ClearValue {
            color: vk::ClearColorValue {
                float32: c.to_owned(),
            },
        },
        ClearValue::ColorU32(c) => vk::ClearValue {
            color: vk::ClearColorValue { uint32: c.to_owned() },
        },
        ClearValue::Depth(d) => vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: *d,
                stencil: 0,
            },
        },
    }
}

impl RenderPass {
    pub fn create_framebuffer(
        self: &Arc<Self>,
        size: (u32, u32),
        attachment_mods: &[(u32, ImageMod)],
    ) -> Result<Arc<Framebuffer>, DeviceError> {
        let attachment_mods: HashMap<u32, ImageMod> = attachment_mods.iter().cloned().collect();
        let mut images = Vec::with_capacity(self.attachments.len());
        let mut native_image_views = Vec::with_capacity(self.attachments.len());

        for (i, attachment) in self.attachments.iter().enumerate() {
            let mut usage = if attachment.format == DEPTH_FORMAT {
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                ImageUsageFlags::COLOR_ATTACHMENT
            };
            let mut override_image = None;

            if let Some(attachment_mod) = attachment_mods.get(&(i as u32)) {
                match attachment_mod {
                    ImageMod::OverrideImage(i) => {
                        override_image = Some(i);
                    }
                    ImageMod::AdditionalUsage(u) => {
                        usage |= *u;
                    }
                }
            }

            let image = if let Some(override_image) = override_image {
                Arc::clone(override_image)
            } else {
                self.device
                    .create_image_2d(attachment.format, 1, 1f32, usage, size)?
            };

            native_image_views.push(image.view);
            images.push(image);
        }

        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(self.native)
            .attachments(&native_image_views)
            .width(size.0)
            .height(size.1)
            .layers(1);

        Ok(Arc::new(Framebuffer {
            device: Arc::clone(&self.device),
            _render_pass: Arc::clone(self),
            native: unsafe { self.device.wrapper.0.create_framebuffer(&create_info, None)? },
            images,
            size,
        }))
    }
}

impl Eq for RenderPass {}

impl PartialEq for RenderPass {
    fn eq(&self, other: &Self) -> bool {
        (self as *const Self) == (other as *const Self)
    }
}

impl Hash for RenderPass {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_usize(self as *const Self as usize);
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_render_pass(self.native, None) };
    }
}
