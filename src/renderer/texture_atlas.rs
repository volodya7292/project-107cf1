use crate::resource_file::ResourceRef;
use crate::utils;
use std::mem;
use std::sync::{Arc, Mutex};
use vk_wrapper as vkw;
use vk_wrapper::ImageLayout;

#[derive(Debug)]
pub enum Error {
    IndexOutOfBounds(String),
}

pub struct TextureAtlas {
    device: Arc<vkw::Device>,
    cmd_list: Arc<Mutex<vkw::CmdList>>,
    submit_packet: vkw::SubmitPacket,
    image: Arc<vkw::Image>,
    size: u32,
    size_in_tiles: u32,
    texture_size: u32,
}

impl TextureAtlas {
    pub fn image(&self) -> Arc<vkw::Image> {
        Arc::clone(&self.image)
    }

    pub fn max_texture_count(&self) -> u32 {
        self.size_in_tiles * self.size_in_tiles
    }

    pub fn set_texture(&mut self, index: u32, mip_maps: &[Vec<u8>]) -> Result<(), Error> {
        let max_index = self.size_in_tiles * self.size_in_tiles - 1;
        if index > max_index {
            return Err(Error::IndexOutOfBounds(format!(
                "index {} > {}",
                index, max_index
            )));
        }

        // Create staging buffer
        let buffer_size =
            (self.texture_size * self.texture_size * vkw::FORMAT_SIZES[&self.image.format()] as u32 * 2)
                as u64;
        let mut buffer = self
            .device
            .create_host_buffer::<u8>(vkw::BufferUsageFlags::TRANSFER_SRC, buffer_size)
            .unwrap();

        // Copy mip_maps
        let mut offset = 0;
        for (level, mip_map) in mip_maps.iter().enumerate() {
            let mip_size = self.calc_texture_size(level as u32);
            let texture_byte_size = mip_size * mip_size * vkw::FORMAT_SIZES[&self.image.format()] as u32;

            buffer.write(offset, &mip_map[..(texture_byte_size as usize)]);
            offset += texture_byte_size as u64;
        }

        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);

        {
            let mut cl = self.cmd_list.lock().unwrap();
            cl.begin(true).unwrap();
            cl.barrier_image(
                vkw::PipelineStageFlags::TOP_OF_PIPE,
                vkw::PipelineStageFlags::TRANSFER,
                &[self.image.barrier_queue(
                    vkw::AccessFlags::default(),
                    vkw::AccessFlags::TRANSFER_WRITE,
                    vkw::ImageLayout::SHADER_READ,
                    vkw::ImageLayout::TRANSFER_DST,
                    graphics_queue,
                    graphics_queue,
                )],
            );

            let mut offset = 0;
            for level in 0..mip_maps.len().min(self.image.mip_levels() as usize) {
                let mip_size = self.calc_texture_size(level as u32);
                let texture_offset = self.get_image_texture_offset(index, level as u32);
                let texture_byte_size = mip_size * mip_size * vkw::FORMAT_SIZES[&self.image.format()] as u32;

                cl.copy_host_buffer_to_image_2d(
                    &buffer,
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
                &[self.image.barrier_queue(
                    vkw::AccessFlags::TRANSFER_WRITE,
                    vkw::AccessFlags::default(),
                    vkw::ImageLayout::TRANSFER_DST,
                    vkw::ImageLayout::SHADER_READ,
                    graphics_queue,
                    graphics_queue,
                )],
            );

            cl.end().unwrap()
        }

        graphics_queue.submit(&mut self.submit_packet).unwrap();
        self.submit_packet.wait().unwrap();
        Ok(())
    }

    fn get_image_texture_offset(&self, index: u32, level: u32) -> (u32, u32) {
        (
            (index % self.size_in_tiles) * self.texture_size / (1 << level),
            (index / self.size_in_tiles) * self.texture_size / (1 << level),
        )
    }

    fn calc_texture_size(&self, level: u32) -> u32 {
        self.texture_size / (1 << level)
    }

    /*pub fn get_cell(&self, pos: (u32, u32)) -> &Cell {
        &self.cells[(pos.1 * self.size_in_cells + pos.0) as usize]
    }*/
}

/// Creates a new texture atlas with resolution (size x size) and
/// max texture resolution (max_texture_size x max_texture_size)
pub fn new(
    device: &Arc<vkw::Device>,
    format: vkw::Format,
    mipmaps: bool,
    max_anisotropy: f32,
    tile_count: u32,
    texture_size: u32,
) -> Result<TextureAtlas, vkw::DeviceError> {
    let max_texture_size = utils::next_power_of_two(texture_size);
    let size_in_tiles = (tile_count as f64).sqrt().ceil() as u32;
    let size = size_in_tiles * max_texture_size;
    let max_mip_levels = if mipmaps {
        (utils::log2(max_texture_size).max(3) - 2) // Account for BC block size (4x4)
    } else {
        1
    };

    let image = device.create_image_2d(
        format,
        max_mip_levels,
        max_anisotropy,
        vkw::ImageUsageFlags::TRANSFER_DST | vkw::ImageUsageFlags::SAMPLED,
        (size, size),
    )?;

    let graphics_queue = device.get_queue(vkw::Queue::TYPE_GRAPHICS);
    let cmd_list = graphics_queue.create_primary_cmd_list()?;

    let mut submit_packet =
        device.create_submit_packet(&[vkw::SubmitInfo::new(&[], &[Arc::clone(&cmd_list)])])?;

    // Change image initial layout
    {
        let mut cl = cmd_list.lock().unwrap();
        cl.begin(true)?;
        cl.barrier_image(
            vkw::PipelineStageFlags::TOP_OF_PIPE,
            vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
            &[image.barrier_queue(
                vkw::AccessFlags::default(),
                vkw::AccessFlags::default(),
                vkw::ImageLayout::UNDEFINED,
                vkw::ImageLayout::SHADER_READ,
                graphics_queue,
                graphics_queue,
            )],
        );
        cl.end()?;
    }

    graphics_queue.submit(&mut submit_packet)?;
    submit_packet.wait()?;

    Ok(TextureAtlas {
        device: Arc::clone(&device),
        cmd_list,
        submit_packet,
        image,
        size,
        size_in_tiles,
        texture_size: max_texture_size,
    })
}
