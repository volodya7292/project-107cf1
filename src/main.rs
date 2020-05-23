use vk_engine::{AccessFlags, CmdList, Format, ImageLayout, PipelineStageFlags, Queue, SubmitInfo};
use winit::window::WindowBuilder;

use std::mem::swap;
use std::rc::Rc;
use vk_engine::image::ImageUsageFlags;
use vk_engine::queue::WaitSemaphore;
use winit::dpi;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

async fn shit() {}

fn main() {
    simple_logger::init().unwrap();

    let s = shit();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("GOVNO!")
        .with_resizable(true)
        .with_visible(true)
        .with_inner_size(dpi::PhysicalSize::new(1280, 720))
        .build(&event_loop)
        .unwrap();

    let windows_extensions = vk_engine::instance::enumerate_required_window_extensions(&window).unwrap();
    let vke = vk_engine::Entry::new().unwrap();
    let instance = vke
        .create_instance("GOVNO!", windows_extensions.iter().map(String::as_str).collect())
        .unwrap();

    let surface = instance.create_surface(&window).unwrap();

    let adapters = instance.enumerate_adapters(&surface).unwrap();
    let mut device = adapters[0].create_device().unwrap();

    //let graph = device.get_queue(Queue::TYPE_GRAPHICS);

    //adapters.iter_mut()

    let mut buf = device
        .create_host_buffer::<u32>(vk_engine::buffer::BufferUsageFlags::UNIFORM, 5)
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

    let mut window_size = window.inner_size();
    let mut swapchain = device
        .create_swapchain(&surface, (window_size.width, window_size.height), true)
        .unwrap();

    let image = device
        .create_image_2d(
            Format::RGBA16_UNORM,
            false,
            ImageUsageFlags::SAMPLED,
            (1024, 1024),
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

    let mut surface_changed = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                println!("The close button was pressed; stopping");
                *control_flow = ControlFlow::Exit;
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(new_size),
                ..
            } => {
                surface_changed = true;
                window_size = new_size;
            }
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                if surface_changed {
                    swapchain = device
                        .create_swapchain(&surface, (window_size.width, window_size.height), true)
                        .unwrap();
                }

                let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);
                let (sw_image, optimal) = swapchain.acquire_image().unwrap();
                surface_changed = !optimal;

                cmd_list.begin(true).unwrap();
                cmd_list.barrier_image(
                    PipelineStageFlags::TOP_OF_PIPE,
                    PipelineStageFlags::BOTTOM_OF_PIPE,
                    &[sw_image.get_image().barrier_queue_level(
                        AccessFlags::empty(),
                        AccessFlags::empty(),
                        ImageLayout::UNDEFINED,
                        ImageLayout::PRESENT,
                        graphics_queue,
                        graphics_queue,
                        0,
                        1,
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
                let optimal = present_queue
                    .present(
                        sw_image,
                        graphics_queue.get_semaphore(),
                        submit_packet.get_signal_value(0),
                        &dummy_cmd_list,
                    )
                    .unwrap();
                surface_changed = !optimal;

                submit_packet.wait().unwrap();
            }
            _ => (),
        }
    });
}
