use crate::module::main_renderer::gpu_executor::{GPUJob, GPUJobDeviceExt, GPUJobExecInfo};
use std::sync::Arc;
use vk_wrapper as vkw;
use vk_wrapper::QueueType;
use vk_wrapper::image::ImageParams;
use vkw::buffer::BufferHandleImpl;

#[derive(Debug)]
pub enum Error {
    IndexOutOfBounds(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IndexOutOfBounds(msg) => write!(f, "Index out of bounds: {}", msg),
        }
    }
}

pub struct TextureAtlas {
    device: Arc<vkw::Device>,
    gpu_job: GPUJob,
    image: Arc<vkw::Image>,
    _width: u32,
    width_in_tiles: u32,
    tile_width: u32,
}

impl TextureAtlas {
    /// Creates a new texture atlas with resolution (size x size) and
    /// max texture resolution (max_texture_size x max_texture_size)
    pub fn new(
        device: &Arc<vkw::Device>,
        format: vkw::Format,
        mipmaps: bool,
        tile_count: u32,
        tile_width: u32,
    ) -> Result<TextureAtlas, vkw::DeviceError> {
        let max_tile_width = tile_width.next_power_of_two();
        let width_in_tiles = (tile_count as f64).sqrt().ceil() as u32;
        let width = width_in_tiles * max_tile_width;
        let max_mip_levels = if mipmaps {
            max_tile_width.ilog2().max(3) - 2 // Account for BC block size (4x4)
        } else {
            1
        };

        let image = device.create_image(
            &ImageParams::d2(
                format,
                vkw::ImageUsageFlags::TRANSFER_DST | vkw::ImageUsageFlags::SAMPLED,
                (width, width),
            )
            .with_preferred_mip_levels(max_mip_levels),
            "tex-atlas",
        )?;

        let mut gpu_job = device.create_job("tex-atlas", QueueType::Graphics)?;

        // Change image initial layout
        {
            let cl = gpu_job.get_cmd_list_for_recording();
            cl.begin(true)?;
            cl.barrier_image(
                vkw::PipelineStageFlags::TOP_OF_PIPE,
                vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                &[image
                    .barrier()
                    .old_layout(vkw::ImageLayout::UNDEFINED)
                    .new_layout(vkw::ImageLayout::SHADER_READ)],
            );
            cl.end()?;
        }

        unsafe {
            device.run_jobs(&mut [GPUJobExecInfo::new(&mut gpu_job)])?;
        }
        gpu_job.wait()?;

        Ok(TextureAtlas {
            device: Arc::clone(device),
            gpu_job,
            image,
            _width: width,
            width_in_tiles,
            tile_width: max_tile_width,
        })
    }

    pub fn image(&self) -> Arc<vkw::Image> {
        Arc::clone(&self.image)
    }

    pub fn width_in_tiles(&self) -> u32 {
        self.width_in_tiles
    }

    pub fn tile_width(&self) -> u32 {
        self.tile_width
    }

    pub fn max_texture_count(&self) -> u32 {
        self.width_in_tiles * self.width_in_tiles
    }

    pub fn set_texture(&mut self, index: u32, mip_maps: &[Vec<u8>]) -> Result<(), Error> {
        let max_index = self.width_in_tiles * self.width_in_tiles - 1;
        if index > max_index {
            return Err(Error::IndexOutOfBounds(format!(
                "index {} > {}",
                index, max_index
            )));
        }

        // Create staging buffer
        let buffer_size =
            (self.tile_width * self.tile_width * vkw::FORMAT_SIZES[&self.image.format()] as u32 * 2) as u64;
        let mut buffer = self
            .device
            .create_host_buffer::<u8>(vkw::BufferUsageFlags::TRANSFER_SRC, buffer_size)
            .unwrap();

        // Copy mip_maps
        let mut offset = 0;
        for (level, mip_map) in mip_maps.iter().enumerate() {
            let mip_size = calc_texture_size(self.tile_width, level as u32);
            let texture_byte_size = mip_size * mip_size * vkw::FORMAT_SIZES[&self.image.format()] as u32;

            buffer.write(offset, &mip_map[..(texture_byte_size as usize)]);
            offset += texture_byte_size as u64;
        }

        {
            let cl = self.gpu_job.get_cmd_list_for_recording();
            cl.begin(true).unwrap();
            cl.barrier_image(
                vkw::PipelineStageFlags::TOP_OF_PIPE,
                vkw::PipelineStageFlags::TRANSFER,
                &[self
                    .image
                    .barrier()
                    .dst_access_mask(vkw::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vkw::ImageLayout::SHADER_READ)
                    .new_layout(vkw::ImageLayout::TRANSFER_DST)],
            );

            let mut offset = 0;
            for level in 0..mip_maps.len().min(self.image.mip_levels() as usize) {
                let mip_size = calc_texture_size(self.tile_width, level as u32);
                let texture_offset =
                    calc_texture_offset(self.width_in_tiles, self.tile_width, index, level as u32);
                let texture_byte_size = mip_size * mip_size * vkw::FORMAT_SIZES[&self.image.format()] as u32;

                cl.copy_host_buffer_to_image_2d(
                    buffer.handle(),
                    offset,
                    &self.image,
                    vkw::ImageLayout::TRANSFER_DST,
                    (texture_offset.0, texture_offset.1),
                    level as u32,
                    (mip_size, mip_size),
                );
                offset += texture_byte_size as u64;
            }

            cl.barrier_image(
                vkw::PipelineStageFlags::TRANSFER,
                vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                &[self
                    .image
                    .barrier()
                    .src_access_mask(vkw::AccessFlags::TRANSFER_WRITE)
                    .old_layout(vkw::ImageLayout::TRANSFER_DST)
                    .new_layout(vkw::ImageLayout::SHADER_READ)],
            );

            cl.end().unwrap()
        }

        unsafe {
            self.device
                .run_jobs_sync(&mut [GPUJobExecInfo::new(&mut self.gpu_job)])
                .unwrap();
        }
        Ok(())
    }

    /*pub fn get_cell(&self, pos: (u32, u32)) -> &Cell {
        &self.cells[(pos.1 * self.size_in_cells + pos.0) as usize]
    }*/
}

fn calc_texture_offset(width_in_tiles: u32, tile_width: u32, index: u32, level: u32) -> (u32, u32) {
    (
        (index % width_in_tiles) * tile_width / (1 << level),
        (index / width_in_tiles) * tile_width / (1 << level),
    )
}

fn calc_texture_size(tile_width: u32, level: u32) -> u32 {
    tile_width / (1 << level)
}
