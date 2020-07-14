use crate::resource_file::ResourceRef;
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

    pub fn set_texture(&mut self, index: u32, bytes: &[u8], format: vkw::Format) -> Result<(), Error> {
        let max_index = self.size_in_tiles * self.size_in_tiles - 1;
        if index > max_index {
            return Err(Error::IndexOutOfBounds(format!(
                "index {} > {}",
                index, max_index
            )));
        }

        // Create staging buffer
        let buffer_size = (self.texture_size * self.texture_size * vkw::FORMAT_SIZES[&format] as u32) as u64;
        let mut buffer = self
            .device
            .create_host_buffer::<u8>(vkw::BufferUsageFlags::TRANSFER_SRC, buffer_size)
            .unwrap();
        buffer.write(0, &bytes[..(buffer_size as usize)]);

        // Create staging image
        let staging_image = self
            .device
            .create_image_2d(
                format,
                self.image.mip_levels(),
                1.0,
                vkw::ImageUsageFlags::TRANSFER_SRC | vkw::ImageUsageFlags::TRANSFER_DST,
                (self.texture_size, self.texture_size),
            )
            .unwrap();
        let mip_levels = staging_image.mip_levels();

        let graphics_queue = self.device.get_queue(vkw::Queue::TYPE_GRAPHICS);

        {
            let mut cl = self.cmd_list.lock().unwrap();
            cl.begin(true).unwrap();
            cl.barrier_image(
                vkw::PipelineStageFlags::TOP_OF_PIPE,
                vkw::PipelineStageFlags::TRANSFER,
                &[staging_image.barrier_queue(
                    vkw::AccessFlags::default(),
                    vkw::AccessFlags::TRANSFER_WRITE,
                    vkw::ImageLayout::UNDEFINED,
                    vkw::ImageLayout::TRANSFER_DST,
                    graphics_queue,
                    graphics_queue,
                )],
            );
            cl.copy_host_buffer_to_image(&buffer, 0, &staging_image, vkw::ImageLayout::TRANSFER_DST);
            /*cl.barrier_image(
                vkw::PipelineStageFlags::TRANSFER,
                vkw::PipelineStageFlags::TRANSFER,
                &[
                    staging_image.barrier_queue(
                        vkw::AccessFlags::TRANSFER_WRITE,
                        vkw::AccessFlags::TRANSFER_READ,
                        vkw::ImageLayout::TRANSFER_DST,
                        vkw::ImageLayout::TRANSFER_SRC,
                        graphics_queue,
                        graphics_queue,
                    ),
                    /*self.image.barrier_queue(
                        vkw::AccessFlags::default(),
                        vkw::AccessFlags::TRANSFER_WRITE,
                        vkw::ImageLayout::SHADER_READ,
                        vkw::ImageLayout::TRANSFER_DST,
                        graphics_queue,
                        graphics_queue,
                    ),*/
                ],
            );*/

            /*let dst_offset = self.get_image_texture_offset(index, 0);
            cl.blit_image_2d(
                &staging_image,
                vkw::ImageLayout::TRANSFER_SRC,
                (0, 0),
                (width, height),
                0,
                &self.image,
                vkw::ImageLayout::TRANSFER_DST,
                dst_offset,
                (self.texture_size, self.texture_size),
                0,
            );*/

            // Generate mipmaps
            // ---------------------------------------------------------------------------------------------------------
            for level in 0..(mip_levels - 1) {
                cl.barrier_image(
                    vkw::PipelineStageFlags::TRANSFER,
                    vkw::PipelineStageFlags::TRANSFER,
                    &[staging_image.barrier_queue_level(
                        vkw::AccessFlags::TRANSFER_WRITE,
                        vkw::AccessFlags::TRANSFER_READ,
                        vkw::ImageLayout::TRANSFER_DST,
                        vkw::ImageLayout::TRANSFER_SRC,
                        graphics_queue,
                        graphics_queue,
                        level,
                        1,
                    )],
                );

                //let src_offset = self.get_image_texture_offset(index, level);
                let src_size = self.calc_texture_size(level);
                //let dst_offset = self.get_image_texture_offset(index, level + 1);
                let dst_size = self.calc_texture_size(level + 1);

                cl.blit_image_2d(
                    &staging_image,
                    vkw::ImageLayout::TRANSFER_SRC,
                    (0, 0),
                    (src_size, src_size),
                    level,
                    &staging_image,
                    vkw::ImageLayout::TRANSFER_DST,
                    (0, 0),
                    (dst_size, dst_size),
                    level + 1,
                );

                /*cl.barrier_image(
                    vkw::PipelineStageFlags::TRANSFER,
                    vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                    &[staging_image.barrier_queue_level(
                        vkw::AccessFlags::TRANSFER_READ,
                        vkw::AccessFlags::default(),
                        vkw::ImageLayout::TRANSFER_SRC,
                        vkw::ImageLayout::SHADER_READ,
                        graphics_queue,
                        graphics_queue,
                        level,
                        1,
                    )],
                );*/
            }

            cl.barrier_image(
                vkw::PipelineStageFlags::TRANSFER,
                vkw::PipelineStageFlags::TRANSFER,
                &[staging_image.barrier_queue_level(
                    vkw::AccessFlags::TRANSFER_WRITE,
                    vkw::AccessFlags::TRANSFER_READ,
                    vkw::ImageLayout::TRANSFER_DST,
                    vkw::ImageLayout::TRANSFER_SRC,
                    graphics_queue,
                    graphics_queue,
                    mip_levels - 1,
                    1,
                )],
            );

            /*cl.barrier_image(
                vkw::PipelineStageFlags::TRANSFER,
                vkw::PipelineStageFlags::BOTTOM_OF_PIPE,
                &[staging_image.barrier_queue_level(
                    vkw::AccessFlags::TRANSFER_WRITE,
                    vkw::AccessFlags::default(),
                    vkw::ImageLayout::TRANSFER_DST,
                    vkw::ImageLayout::SHADER_READ,
                    graphics_queue,
                    graphics_queue,
                    mip_levels - 1,
                    1,
                )],
            );*/

            // Copy mipmaps to main image
            // ---------------------------------------------------------------------------------------------------------
            cl.barrier_image(
                vkw::PipelineStageFlags::TRANSFER,
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

            for level in 0..mip_levels {
                let dst_offset = self.get_image_texture_offset(index, level);
                let size = self.calc_texture_size(level);

                cl.copy_image_2d(
                    &staging_image,
                    vkw::ImageLayout::TRANSFER_SRC,
                    (0, 0),
                    level,
                    &self.image,
                    vkw::ImageLayout::TRANSFER_DST,
                    dst_offset,
                    level,
                    (size / 4, size / 4), // divide by compressed block size
                );
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

fn next_power_of_two(mut n: u32) -> u32 {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n += 1;
    n
}

fn make_mul_of(n: u32, m: u32) -> u32 {
    ((n + m - 1) / m) * m
}

const fn num_bits<T>() -> usize {
    std::mem::size_of::<T>() * 8
}

fn log2(n: u32) -> u32 {
    (mem::size_of::<u32>() * 8) as u32 - n.leading_zeros() - 1
}

/// Creates a new texture atlas with resolution (size x size) and
/// max texture resolution (max_texture_size x max_texture_size)
pub fn new(
    device: &Arc<vkw::Device>,
    format: vkw::Format,
    gen_mipmaps: bool,
    max_anisotropy: f32,
    tile_count: u32,
    texture_size: u32,
) -> Result<TextureAtlas, vkw::DeviceError> {
    let max_texture_size = next_power_of_two(texture_size);
    let size_in_tiles = (tile_count as f64).sqrt().ceil() as u32;
    let size = size_in_tiles * max_texture_size;
    let max_mip_levels = if gen_mipmaps { log2(max_texture_size) } else { 1 };

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
