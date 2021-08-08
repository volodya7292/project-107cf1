use std::path::Path;
use std::time::Instant;

use nalgebra_glm as glm;
use utils::thread_pool::ThreadPool;
use winit::dpi::PhysicalPosition;
use winit::event::{VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::platform::run_return::EventLoopExtRunReturn;
use winit::window::{Fullscreen, WindowBuilder};

use crate::render_engine::vertex_mesh::VertexMeshCreate;
use crate::render_engine::{component, TextureQuality, TranslucencyMaxDepth};
use crate::render_engine::{material_pipeline, material_pipelines};
use crate::resource_file::ResourceFile;
use crate::utils::noise::ParamNoise;
use nalgebra_glm::{DVec2, Vec2, Vec3};
use noise::Seedable;

mod utils;

#[macro_use]
pub(crate) mod render_engine;
mod game;
mod resource_file;

#[cfg(test)]
mod tests;

#[derive(Copy, Clone, Default)]
pub struct BasicVertex {
    position: Vec3,
    tex_coord: Vec2,
}

vertex_impl!(BasicVertex, position, tex_coord);

const PROGRAM_NAME: &str = "project-107cf1";
const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);

fn noise_test() {
    let mut buf = vec![0_u8; 1024 * 1024 * 3];

    let n = noise::SuperSimplex::new().set_seed(0);

    let process = |p: DVec2| -> f64 {
        let v = n.sample(p, 5.0, 1.0, 0.5);

        let d = (glm::distance(&DVec2::new(0.5, 0.5), &p) * 2.0).min(1.0);
        let s = glm::smoothstep(0.0, 1.0, 4.0 * (1.0 - d));

        // y = 0.4 - (0.1 - (x - 0.4)) / 0.1

        v * s
    };

    for x in 0..1024 {
        for y in 0..1024 {
            let i = (y * 1024 + x) * 3;
            let x = x as f64 / 1024.0;
            let y = y as f64 / 1024.0;

            let v = process(DVec2::new(x, y));

            buf[i] = (v * 255.0) as u8;
            buf[i + 1] = buf[i];
            buf[i + 2] = buf[i];
        }
    }

    image::save_buffer("D:/noise_test.png", &buf, 1024, 1024, image::ColorType::Rgb8).unwrap();
    std::process::exit(0);
}

fn main() {
    simple_logger::SimpleLogger::new().init().unwrap();

    let thread_count = num_cpus::get_physical().max(2);
    // Note: use safe thread pools to account for proper destruction of Vulkan objects.
    let render_thread_pool = ThreadPool::new(thread_count / 2).unwrap();
    let update_thread_pool = ThreadPool::new(thread_count / 2).unwrap();

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

    let window_size = window.inner_size();

    let renderer_settings = render_engine::Settings {
        vsync: true,
        texture_quality: TextureQuality::STANDARD,
        translucency_max_depth: TranslucencyMaxDepth::LOW,
        textures_gen_mipmaps: true,
        textures_max_anisotropy: 1.0,
    };
    let renderer = render_engine::new(
        &surface,
        (window_size.width, window_size.height),
        renderer_settings,
        &device,
        &mut resources,
        4,
    );

    let mat_pipelines;
    {
        let mut renderer = renderer.lock().unwrap();
        let index = renderer.add_texture(
            render_engine::TextureAtlasType::ALBEDO,
            resources.get("textures/test_texture.ktx").unwrap(),
        );

        renderer.load_texture(index);

        mat_pipelines = material_pipelines::create(&resources, &mut renderer);
    }

    let mut program = game::new(&renderer, &mat_pipelines);

    let triangle_mesh = device
        .create_vertex_mesh::<BasicVertex>(
            &[
                BasicVertex {
                    position: Vec3::new(0.0, -0.5, -3.0),
                    tex_coord: Vec2::new(1.0, 2.0),
                },
                BasicVertex {
                    position: Vec3::new(-0.5, 0.5, -3.0),
                    tex_coord: Vec2::new(0.0, 0.0),
                },
                BasicVertex {
                    position: Vec3::new(0.5, 0.5, -3.0),
                    tex_coord: Vec2::new(2.0, 0.0),
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
            component::Renderer::new(&renderer, mat_pipelines.triag(), false),
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
        let mut comps = render_engine.scene().world.write_component::<component::Transform>();
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

    let mut cursor_grab = true;
    let mut start_t = Instant::now();
    let mut delta_time = 0.0;

    window.set_cursor_grab(cursor_grab).unwrap();
    window.set_cursor_visible(!cursor_grab);

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
                            VirtualKeyCode::T => {
                                if input.state == ElementState::Released {
                                    cursor_grab = !cursor_grab;
                                    window.set_cursor_grab(cursor_grab).unwrap();
                                    window.set_cursor_visible(!cursor_grab);
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
                program.on_update(delta_time, &update_thread_pool);
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                render_thread_pool.install(|| renderer.lock().unwrap().on_draw());
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
