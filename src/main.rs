mod utils;
mod world;
#[macro_use]
pub(crate) mod renderer;
mod object;
mod program;
mod resource_file;
#[cfg(test)]
mod tests;

use crate::renderer::vertex_mesh::VertexMeshCreate;
use crate::renderer::{component, TextureQuality, TranslucencyMaxDepth};
use crate::renderer::{material_pipeline, material_pipelines};
use crate::resource_file::ResourceFile;
use na::Vector2;
use na::Vector3;
use nalgebra as na;
use sdl2::keyboard::Keycode;
use std::path::Path;
use std::time::Instant;

#[derive(Copy, Clone, Default)]
pub struct BasicVertex {
    position: na::Vector3<f32>,
    tex_coord: na::Vector2<f32>,
}

vertex_impl!(BasicVertex, position, tex_coord);

const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();

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
    sdl_context.mouse().set_relative_mouse_mode(true); // enable relative-pos events
    sdl_context.mouse().show_cursor(false);

    let windows_extensions = window.vulkan_instance_extensions().unwrap();
    let vke = vk_wrapper::Entry::new().unwrap();
    let instance = vke.create_instance("GOVNO!", &windows_extensions).unwrap();

    let surface = instance.create_surface(&window).unwrap();

    let adapters = instance.enumerate_adapters(Some(&surface)).unwrap();
    let adapter = adapters.first().unwrap();
    let device = adapter.create_device().unwrap();

    let mut window_size = window.vulkan_drawable_size();

    let renderer_settings = renderer::Settings {
        vsync: true,
        texture_quality: TextureQuality::STANDARD,
        translucency_max_depth: TranslucencyMaxDepth::LOW,
        textures_gen_mipmaps: true,
        textures_max_anisotropy: 1.0,
    };
    let renderer = renderer::new(
        &surface,
        window_size,
        renderer_settings,
        &device,
        &mut resources,
        4,
    )
    .unwrap();

    {
        let mut renderer = renderer.lock().unwrap();
        let index = renderer.add_texture(
            renderer::TextureAtlasType::ALBEDO,
            resources.get("textures/test_texture.ktx").unwrap(),
        );

        renderer.load_texture(index);
    }

    let mat_pipelines = material_pipelines::create(&resources, &device);

    let mut program = program::new(&renderer, &mat_pipelines);

    let triangle_mesh = device
        .create_vertex_mesh::<BasicVertex>(
            &[
                BasicVertex {
                    position: Vector3::new(0.0, -0.5, -3.0),
                    tex_coord: Vector2::new(1.0, 2.0),
                },
                BasicVertex {
                    position: Vector3::new(-0.5, 0.5, -3.0),
                    tex_coord: Vector2::new(0.0, 0.0),
                },
                BasicVertex {
                    position: Vector3::new(0.5, 0.5, -3.0),
                    tex_coord: Vector2::new(2.0, 0.0),
                },
            ],
            None,
        )
        .unwrap();

    {
        let mut renderer = renderer.lock().unwrap();
        let scene = renderer.scene();
        let mut entities = scene.entities().lock().unwrap();
        let transform_comps = scene.storage::<component::Transform>();
        let mut transform_comps = transform_comps.write().unwrap();
        let renderer_comps = scene.storage::<component::Renderer>();
        let mut renderer_comps = renderer_comps.write().unwrap();
        let vertex_mesh_comps = scene.storage::<component::VertexMesh>();
        let mut vertex_mesh_comps = vertex_mesh_comps.write().unwrap();

        let ent0 = entities.create();

        transform_comps.set(ent0, component::Transform::default());
        renderer_comps.set(
            ent0,
            component::Renderer::new(&device, &mat_pipelines.triag(), false),
        );
        vertex_mesh_comps.set(ent0, component::VertexMesh::new(&triangle_mesh.raw()));

        let ent1 = entities.create();

        transform_comps.set(
            ent0,
            component::Transform::new(
                na::Vector3::new(0.0, 0.0, 1.0),
                na::Vector3::default(),
                na::Vector3::new(1.0, 1.0, 1.0),
            ),
        );
        renderer_comps.set(
            ent0,
            component::Renderer::new(&device, &mat_pipelines.triag(), false),
        );
        vertex_mesh_comps.set(ent0, component::VertexMesh::new(&triangle_mesh.raw()));
    }

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

    let mut delta_time = 0.0;

    let mut running = true;
    while running {
        let start_t = Instant::now();

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
                        window.set_size(DEF_WINDOW_SIZE.0, DEF_WINDOW_SIZE.1).unwrap();
                        window
                            .set_position(sdl2::video::WindowPos::Centered, sdl2::video::WindowPos::Centered);
                    } else {
                        let curr_mode = sdl_context
                            .video()
                            .unwrap()
                            .current_display_mode(window.display_index().unwrap())
                            .unwrap();
                        window.set_size(curr_mode.w as u32, curr_mode.h as u32).unwrap();
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
                        renderer.lock().unwrap().on_resize(window_size);
                    }
                    _ => {}
                },
                _ => {}
            }

            program.on_event(event);
        }

        program.on_update(delta_time);
        renderer.lock().unwrap().on_draw();

        let end_t = Instant::now();
        let t = end_t.duration_since(start_t);
        delta_time = t.as_secs_f64();
    }
}
