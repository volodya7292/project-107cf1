use crate::{
    device::DeviceError, AccessFlags, Device, Format, Framebuffer, Image, ImageLayout, ImageUsageFlags,
    PipelineStageFlags,
};
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use std::rc::Rc;

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

pub struct AttachmentRef {
    pub index: u32,
    pub layout: ImageLayout,
}

pub struct Subpass<'a> {
    pub color: &'a [AttachmentRef],
    pub depth: Option<AttachmentRef>,
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
    pub(crate) device: Rc<Device>,
    pub(crate) native: vk::RenderPass,
    pub(crate) attachments: Vec<Attachment>,
    pub(crate) color_attachments: Vec<u32>,
    pub(crate) depth_attachments: Vec<u32>,
}

#[derive(Clone)]
pub enum ImageMod {
    OverrideImage(Rc<Image>),
    AdditionalUsage(ImageUsageFlags),
}

pub enum ClearValue {
    ColorF32([f32; 4]),
    Depth(f32),
}

impl RenderPass {
    pub fn create_framebuffer(
        self: &Rc<Self>,
        size: (u32, u32),
        attachment_mods: &[(u32, ImageMod)],
    ) -> Result<Rc<Framebuffer>, DeviceError> {
        let attachment_mods: HashMap<u32, ImageMod> = attachment_mods.iter().cloned().collect();
        let mut images = Vec::with_capacity(self.attachments.len());
        let mut native_image_views = Vec::with_capacity(self.attachments.len());

        macro_rules! process_attachment {
            ($attachment: ident, $attachment_mod: ident, $usage: ident) => {
                let mut override_image = None;

                if let Some($attachment_mod) = attachment_mods.get(&$attachment) {
                    match $attachment_mod {
                        ImageMod::OverrideImage(i) => {
                            override_image = Some(i);
                        }
                        ImageMod::AdditionalUsage(u) => {
                            $usage |= *u;
                        }
                    }
                }

                let image = if let Some(override_image) = override_image {
                    override_image.clone()
                } else {
                    self.device.create_image_2d(
                        self.attachments[*$attachment as usize].format,
                        false,
                        $usage,
                        size,
                    )?
                };

                native_image_views.push(image.view);
                images.push(image);
            };
        }

        for color_attachment in &self.color_attachments {
            let mut usage = ImageUsageFlags::COLOR_ATTACHMENT;
            process_attachment!(color_attachment, attachment_mod, usage);
        }
        for depth_attachment in &self.depth_attachments {
            let mut usage = ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT;
            process_attachment!(depth_attachment, attachment_mod, usage);
        }

        let create_info = vk::FramebufferCreateInfo::builder()
            .render_pass(self.native)
            .attachments(&native_image_views)
            .width(size.0)
            .height(size.1)
            .layers(1);

        Ok(Rc::new(Framebuffer {
            device: Rc::clone(&self.device),
            renderpass: Rc::clone(self),
            native: unsafe { self.device.wrapper.0.create_framebuffer(&create_info, None)? },
            images,
            size,
        }))
    }
}

impl Drop for RenderPass {
    fn drop(&mut self) {
        unsafe { self.device.wrapper.0.destroy_render_pass(self.native, None) };
    }
}
