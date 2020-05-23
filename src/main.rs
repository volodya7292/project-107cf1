use vk_engine::{Format, PrimaryCmdList, Queue};
use winit::window::WindowBuilder;

use vk_engine::image::ImageUsageFlags;
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
    for a in buf.into_iter() {
        println!("{}", a);
    }
    //println!("ITE {:?}", &buf[0..5]);

    let window_size = window.inner_size();
    let swapchain = device
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
    let graphics_queue = device.get_queue(Queue::TYPE_GRAPHICS);
    let cmd_list = graphics_queue.create_primary_cmd_list().unwrap();

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
            Event::MainEventsCleared => {
                // Application update code.

                // Queue a RedrawRequested event.
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                // Redraw the application.
                //
                // It's preferrable to render in this event rather than in MainEventsCleared, since
                // rendering in here allows the program to gracefully handle redraws requested
                // by the OS.

                let (sw_image, success) = swapchain.acquire_image().unwrap();

                //let queue = device.get_queue(Queue::TYPE_PRESENT);
                //queue.present(sw_image, queue.get_semaphore(), 0).unwrap();
                //println!("f");
            }
            _ => (),
        }
    });
}
