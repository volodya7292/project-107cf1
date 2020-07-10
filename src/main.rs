#[macro_use]
mod renderer;
mod resource_file;

use crate::renderer::vertex_mesh::{Vertex, VertexMeshCreate};
use crate::renderer::{component, TextureQuality};
use crate::renderer::{material_pipeline, material_pipelines};
use crate::resource_file::ResourceFile;
use nalgebra::Vector3;
use sdl2::keyboard::Keycode;
use sdl2::pixels::PixelFormatEnum;
use sdl2::rect::Rect;
use specs::prelude::*;
use specs::{Builder, WorldExt};
use std::path::Path;
use vk_wrapper as vkw;
use vk_wrapper::{HostBuffer, PrimitiveTopology};
use vk_wrapper::{ImageUsageFlags, Subpass};

#[derive(Default)]
pub struct BasicVertex {
    position: nalgebra::Vector3<f32>,
}

vertex_impl!(BasicVertex, position);

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

    let renderer_settings = renderer::Settings {
        vsync: true,
        texture_quality: TextureQuality::LOW,
        textures_gen_mipmaps: true,
        textures_max_anisotropy: 1.0,
    };
    let mut renderer =
        renderer::new(&surface, window_size, renderer_settings, &device, &mut resources).unwrap();

    let index = renderer.add_texture(
        resources.get("textures/test_texture.jpg").unwrap(),
        renderer::TextureAtlasType::ALBEDO,
    );
    renderer.load_texture(index);

    let mat_pipelines = material_pipelines::create(&device);

    let mut triangle_mesh = device.create_vertex_mesh::<BasicVertex>().unwrap();
    triangle_mesh.set_vertices(
        &[
            BasicVertex {
                position: Vector3::new(0.0, -0.5, -3.0),
            },
            BasicVertex {
                position: Vector3::new(-0.5, 0.5, -3.0),
            },
            BasicVertex {
                position: Vector3::new(0.5, 0.5, -3.0),
            },
        ],
        &[],
    );

    let entity = renderer
        .add_entity()
        .with(component::Transform::default())
        .with(component::VertexMeshRef::new(
            triangle_mesh.raw().as_ref().unwrap(),
        ))
        .with(component::Renderer::new(&device, &mat_pipelines.basic(), false))
        .build();

    /*{
        let mut comps = renderer.scene().world.write_component::<component::Transform>();
        let mut trans = comps.get_mut(entity).unwrap();
        *trans = component::Transform::default();
    }*/

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
