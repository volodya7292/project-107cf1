use ash::vk;
use common::types::HashMap;
use std::sync::Arc;

use crate::format::DEPTH_FORMAT;
use crate::{
    device::DeviceError, AccessFlags, Device, Format, Framebuffer, Image, ImageLayout, ImageUsageFlags,
    ImageView, PipelineStageFlags,
};

#[derive(Copy, Clone)]
pub enum LoadStore {
    None,
    InitLoad,
    InitClear,
    FinalSave,
    InitClearFinalSave,
    InitLoadFinalSave,
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
    pub input: Vec<AttachmentRef>,
    pub color: Vec<AttachmentRef>,
    pub depth: Option<AttachmentRef>,
}

impl Subpass {
    pub const EXTERNAL: u32 = vk::SUBPASS_EXTERNAL;

    pub fn new() -> Self {
        Self {
            input: vec![],
            color: vec![],
            depth: None,
        }
    }

    pub fn with_input(mut self, att_refs: Vec<AttachmentRef>) -> Self {
        self.input = att_refs;
        self
    }

    pub fn with_color(mut self, att_refs: Vec<AttachmentRef>) -> Self {
        self.color = att_refs;
        self
    }

    pub fn with_depth(mut self, att_ref: AttachmentRef) -> Self {
        self.depth = Some(att_ref);
        self
    }
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
    pub(crate) _input_attachments: Vec<u32>,
}

#[derive(Clone)]
pub enum ImageMod {
    OverrideImage(Arc<Image>),
    OverrideImageView(Arc<ImageView>),
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
        let mut image_views = Vec::with_capacity(self.attachments.len());
        let mut images = Vec::with_capacity(self.attachments.len());
        let mut native_image_views = Vec::with_capacity(self.attachments.len());

        for (i, attachment) in self.attachments.iter().enumerate() {
            let mut usage = if attachment.format == DEPTH_FORMAT {
                ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                ImageUsageFlags::COLOR_ATTACHMENT
            };
            let mut override_image = None;
            let mut override_image_view = None;

            if let Some(attachment_mod) = attachment_mods.get(&(i as u32)) {
                match attachment_mod {
                    ImageMod::OverrideImage(i) => {
                        override_image = Some(Arc::clone(&i));
                        override_image_view = Some(Arc::clone(&i.view));
                    }
                    ImageMod::OverrideImageView(v) => {
                        override_image_view = Some(Arc::clone(v));
                    }
                    ImageMod::AdditionalUsage(u) => {
                        usage |= *u;
                    }
                }
            }

            let (image, image_view) = if let Some(override_image_view) = override_image_view {
                (override_image, override_image_view)
            } else {
                let image = self.device.create_image_2d(attachment.format, 1, usage, size)?;
                let view = Arc::clone(&image.view);
                (Some(image), view)
            };

            native_image_views.push(image_view.native);
            image_views.push(image_view);
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
            native: unsafe {
                self.device
                    .wrapper
                    .native
                    .create_framebuffer(&create_info, None)?
            },
            images,
            _image_views: image_views,
            size,
        }))
    }
}

impl PartialEq for RenderPass {
    fn eq(&self, other: &Self) -> bool {
        self.native == other.native
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.native.destroy_render_pass(self.native, None) };
    }
}
