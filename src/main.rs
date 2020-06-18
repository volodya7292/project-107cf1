mod renderer;
mod resource_file;

use crate::resource_file::ResourceFile;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use specs::Builder;
use std::path::Path;
use vk_wrapper::render_pass::{AttachmentRef, ImageMod};
use vk_wrapper::HostBuffer;
use vk_wrapper::WaitSemaphore;
use vk_wrapper::{
    swapchain, AccessFlags, Attachment, Format, ImageLayout, LoadStore, PipelineStageFlags, Queue, SubmitInfo,
};
use vk_wrapper::{ImageUsageFlags, Subpass};

fn main() {
    simple_logger::init().unwrap();

    let mut resources = ResourceFile::open(Path::new("resources")).unwrap();

    let sdl_context = sdl2::init().unwrap();

    let mut window = sdl_context
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

    let mut window_size = window.vulkan_drawable_size();
    let mut renderer = renderer::new(&surface, window_size, true, &device).unwrap();

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

    //let mut buf2: HostBuffer<u8> = buf;

    // for a in buf.into_iter() {
    //     println!("{}", a);
    // }
    //println!("ITE {:?}", &buf[0..5]);

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
    let shader = device.create_shader(&shader_spv).unwrap();

    let mut scene = renderer.scene();

    scene
        .create_entity()
        .with(renderer::component::Transform::new())
        .build();
    scene
        .create_entity()
        .with(renderer::component::Transform::new())
        .build();

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
                Event::KeyDown {
                    keycode: Some(Keycode::F11),
                    ..
                } => {
                    if window.fullscreen_state() == sdl2::video::FullscreenType::True {
                        window.set_fullscreen(sdl2::video::FullscreenType::Off).unwrap();
                    } else {
                        let mut curr_mode = sdl_context
                            .video()
                            .unwrap()
                            .current_display_mode(window.display_index().unwrap())
                            .unwrap();
                        curr_mode.format = sdl2::pixels::PixelFormatEnum::RGBA32;

                        window.set_display_mode(curr_mode).unwrap();
                        window.set_fullscreen(sdl2::video::FullscreenType::True).unwrap();
                    }
                }
                Event::Window {
                    timestamp: _,
                    window_id: _,
                    win_event,
                } => match win_event {
                    WindowEvent::SizeChanged(width, height) => {
                        window_size = (width as u32, height as u32);
                        renderer.on_resize(window_size);
                    }
                    _ => {}
                },
                _ => {}
            }
        }

        renderer.on_draw();
    }
}
