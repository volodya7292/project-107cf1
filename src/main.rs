mod resource_file;

use crate::resource_file::ResourceFile;
use engine_3d::scene;
use engine_3d::specs::Builder;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use std::path::Path;
use std::rc::Rc;
use vk_wrapper::render_pass::{AttachmentRef, ImageMod};
use vk_wrapper::ShaderBindingType;
use vk_wrapper::WaitSemaphore;
use vk_wrapper::{
    swapchain, AccessFlags, Attachment, Format, ImageLayout, LoadStore, PipelineStageFlags, Queue, SubmitInfo,
};
use vk_wrapper::{ImageUsageFlags, Subpass};

fn main() {
    simple_logger::init().unwrap();

    let mut resources = ResourceFile::open(Path::new("resources")).unwrap();

    let sdl_context = sdl2::init().unwrap();

    let window = sdl_context
        .video()
        .unwrap()
        .window("project-107cf1", 1280, 720)
        .resizable()
        .position_centered()
        .vulkan()
        .build()
        .unwrap();

    let windows_extensions = window.vulkan_instance_extensions().unwrap();
    let vke = vk_wrapper::Entry::new().unwrap();
    let instance = vke.create_instance("GOVNO!", windows_extensions).unwrap();

    let surface = instance.create_surface(&window).unwrap();

    let adapters = instance.enumerate_adapters(&surface).unwrap();
    let adapter = adapters.first().unwrap();
    let device = adapter.create_device().unwrap();

    //let graph = device.get_queue(Queue::TYPE_GRAPHICS);

    //adapters.iter_mut()

    let mut buf = device
        .create_host_buffer::<u32>(vk_wrapper::buffer::BufferUsageFlags::UNIFORM, 5)
        .unwrap();
    buf[0] = 1u32;
    buf[1] = 2u32;
    buf[2] = 3u32;
    buf[3] = 4u32;
    buf[4] = 5u32;
    // for a in buf.into_iter() {
    //     println!("{}", a);
    // }
    //println!("ITE {:?}", &buf[0..5]);

    let mut window_size = window.vulkan_drawable_size();
    let mut swapchain = device.create_swapchain(&surface, window_size, true).unwrap();

    let render_pass = device
        .create_render_pass(
            &[Attachment {
                format: Format::RGBA16_FLOAT,
                init_layout: ImageLayout::UNDEFINED,
                final_layout: ImageLayout::PRESENT,
                load_store: LoadStore::FinalSave,
            }],
            &[Subpass {
                color: &[AttachmentRef {
                    index: 0,
                    layout: ImageLayout::COLOR_ATTACHMENT,
                }],
                depth: None,
            }],
            &[],
        )
        .unwrap();

    let cmd_list = device
        .get_queue(Queue::TYPE_GRAPHICS)
        .create_primary_cmd_list()
        .unwrap();
    let dummy_cmd_list = device
        .get_queue(Queue::TYPE_GRAPHICS)
        .create_primary_cmd_list()
        .unwrap();
    dummy_cmd_list.begin(false).unwrap();
    dummy_cmd_list.end().unwrap();

    let shader_spv = resources.read("shaders/cluster.frag.spv").unwrap();
    let shader = device
        .create_shader(&shader_spv, &[("", ShaderBindingType::DEFAULT)])
        .unwrap();

    let mut d = scene::new();
    let ent = d.create_entity().build();

    let mut surface_changed = false;

    let mut running = true;
    while running {
        for event in sdl_context.event_pump().unwrap().poll_iter() {
            use sdl2::event::Event;
            use sdl2::event::WindowEvent;
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => {
                    running = false;
                }
                Event::Window {
                    timestamp: _,
                    window_id: _,
                    win_event,
                } => match win_event {
                    WindowEvent::Resized(width, height) => {
                        surface_changed = true;
                        window_size = (width as u32, height as u32);
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        if adapter.is_surface_valid(&surface).unwrap() {
            if surface_changed {
                swapchain = device
                    .create_swapchain(&surface, (window_size.0, window_size.1), true)
                    .unwrap();
                surface_changed = false;
            }

            let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);

            let acquire_result = swapchain.acquire_image();
            if let Ok((sw_image, optimal)) = acquire_result {
                surface_changed = !optimal;

                /*let framebuffer = render_pass
                .create_framebuffer(
                    window_size,
                    &[(0, ImageMod::OverrideImage(Rc::clone(&sw_image.get_image())))],
                )
                .unwrap();*/

                cmd_list.begin(true).unwrap();
                cmd_list.barrier_image(
                    PipelineStageFlags::TOP_OF_PIPE,
                    PipelineStageFlags::BOTTOM_OF_PIPE,
                    &[sw_image.get_image().barrier_queue(
                        AccessFlags::empty(),
                        AccessFlags::empty(),
                        ImageLayout::UNDEFINED,
                        ImageLayout::PRESENT,
                        graphics_queue,
                        graphics_queue,
                    )],
                );
                cmd_list.end().unwrap();

                let mut submit_packet = device
                    .create_submit_packet(&[SubmitInfo::new(
                        &[WaitSemaphore {
                            semaphore: swapchain.get_semaphore(),
                            wait_dst_mask: PipelineStageFlags::TOP_OF_PIPE,
                            wait_value: 0,
                        }],
                        &[Rc::clone(&cmd_list)],
                    )])
                    .unwrap();
                graphics_queue.submit(&mut submit_packet).unwrap();

                let present_queue = device.get_queue(Queue::TYPE_PRESENT);
                let result = present_queue.present(
                    sw_image,
                    graphics_queue.get_semaphore(),
                    submit_packet.get_signal_value(0),
                );

                if let Ok(optimal) = result {
                    surface_changed = !optimal;
                } else {
                    surface_changed = true;
                }

                let dd = graphics_queue == present_queue;
            } else {
                match acquire_result {
                    Err(swapchain::Error::IncompatibleSurface) => {
                        surface_changed = true;
                    }
                    _ => {
                        acquire_result.unwrap();
                    }
                }
            }
        }
    }
}
