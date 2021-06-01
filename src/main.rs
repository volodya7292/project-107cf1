mod utils;

#[macro_use]
pub(crate) mod renderer;
mod object;
mod program;
mod resource_file;
mod world;

#[cfg(test)]
mod tests;

use crate::renderer::vertex_mesh::VertexMeshCreate;
use crate::renderer::{component, TextureQuality, TranslucencyMaxDepth};
use crate::renderer::{material_pipeline, material_pipelines};
use crate::resource_file::ResourceFile;
use na::Vector2;
use na::Vector3;
use nalgebra as na;
use raw_window_handle::HasRawWindowHandle;
use std::path::Path;
use std::time::Instant;
use vk_wrapper::PrimitiveTopology;
use winit::dpi::PhysicalPosition;
use winit::event::{VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::monitor::VideoMode;
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Fullscreen, Window, WindowBuilder};

#[derive(Copy, Clone, Default)]
pub struct BasicVertex {
    position: na::Vector3<f32>,
    tex_coord: na::Vector2<f32>,
}

vertex_impl!(BasicVertex, position, tex_coord);

const PROGRAM_NAME: &str = "project-107cf1";
const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();

    let mut resources = ResourceFile::open(Path::new("resources")).unwrap();

    let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title(PROGRAM_NAME)
        .with_inner_size(winit::dpi::PhysicalSize::new(
            DEF_WINDOW_SIZE.0,
            DEF_WINDOW_SIZE.1,
        ))
        .with_resizable(true)
        .build(&event_loop)
        .unwrap();

    // Center the window
    {
        let win_size = window.outer_size();
        let mode = window.current_monitor().unwrap().video_modes().next().unwrap();
        let mon_size = mode.size();
        window.set_outer_position(PhysicalPosition {
            x: (mon_size.width - win_size.width) / 2,
            y: (mon_size.height - win_size.height) / 2,
        });
    }

    let vke = vk_wrapper::Entry::new().unwrap();
    let instance = vke.create_instance(PROGRAM_NAME, &window).unwrap();

    let surface = instance.create_surface(&window).unwrap();

    let adapters = instance.enumerate_adapters(Some(&surface)).unwrap();
    let adapter = adapters.first().unwrap();
    let device = adapter.create_device().unwrap();

    let mut window_size = window.inner_size();

    let renderer_settings = renderer::Settings {
        vsync: true,
        texture_quality: TextureQuality::STANDARD,
        translucency_max_depth: TranslucencyMaxDepth::LOW,
        textures_gen_mipmaps: true,
        textures_max_anisotropy: 1.0,
    };
    let renderer = renderer::new(
        &surface,
        (window_size.width, window_size.height),
        renderer_settings,
        &device,
        &mut resources,
        4,
    )
    .unwrap();

    let mat_pipelines;
    {
        let mut renderer = renderer.lock().unwrap();
        let index = renderer.add_texture(
            renderer::TextureAtlasType::ALBEDO,
            resources.get("textures/test_texture.ktx").unwrap(),
        );

        renderer.load_texture(index);

        mat_pipelines = material_pipelines::create(&resources, &mut renderer);
    }

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
        let renderer = renderer.lock().unwrap();
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

        // let ent1 = entities.create();
        //
        // transform_comps.set(
        //     ent1,
        //     component::Transform::new(
        //         na::Vector3::new(0.0, 0.0, 1.0),
        //         na::Vector3::default(),
        //         na::Vector3::new(1.0, 1.0, 1.0),
        //     ),
        // );
        // renderer_comps.set(
        //     ent1,
        //     component::Renderer::new(&device, &mat_pipelines.triag(), false),
        // );
        // vertex_mesh_comps.set(ent1, component::VertexMesh::new(&triangle_mesh.raw()));
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

    let mut start_t = Instant::now();
    let mut delta_time = 0.0;

    window.set_cursor_grab(true).unwrap();
    window.set_cursor_visible(false);

    event_loop.run_return(|event, _, control_flow| {
        use winit::event::ElementState;
        use winit::event::Event;

        *control_flow = ControlFlow::Poll;

        match event {
            Event::NewEvents(_) => {
                start_t = Instant::now();
            }
            Event::WindowEvent { window_id, ref event } => match event {
                WindowEvent::Resized(size) => {
                    if size.width != 0 && size.height != 0 {
                        renderer.lock().unwrap().on_resize((size.width, size.height));
                    }
                }
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input,
                    is_synthetic: _,
                } => {
                    if let Some(keycode) = input.virtual_keycode {
                        match keycode {
                            VirtualKeyCode::Escape => {
                                *control_flow = ControlFlow::Exit;
                            }
                            VirtualKeyCode::F11 => {
                                if input.state == ElementState::Released {
                                    if let Some(_) = window.fullscreen() {
                                        window.set_fullscreen(None);
                                    } else {
                                        let mode =
                                            window.current_monitor().unwrap().video_modes().next().unwrap();
                                        window.set_fullscreen(Some(Fullscreen::Exclusive(mode)))
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::CloseRequested if window_id == window.id() => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            Event::MainEventsCleared => {
                program.on_update(delta_time);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                renderer.lock().unwrap().on_draw();
            }
            Event::RedrawEventsCleared => {
                let end_t = Instant::now();
                delta_time = (end_t - start_t).as_secs_f64();
            }
            _ => {}
        }

        program.on_event(event);
    });
}
