pub mod main_registry;
pub mod overworld;
pub mod overworld_streamer;
pub mod registry;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::block::Block;
use crate::game::overworld::block_component::Facing;
use crate::game::overworld::block_model;
use crate::game::overworld::block_model::PackedVertex;
use crate::game::overworld::cluster::BlockDataImpl;
use crate::game::overworld::light_level::LightLevel;
use crate::game::overworld_streamer::OverworldStreamer;
use crate::physics::aabb::{AABBRayIntersection, AABB};
use crate::physics::MOTION_EPSILON;
use crate::{material_pipelines, physics, utils, PROGRAM_NAME};
use approx::AbsDiffEq;
use engine::ecs::component;
use engine::ecs::scene::Scene;
use engine::queue::{coroutine_queue, intensive_queue};
use engine::renderer::vertex_mesh::VertexMeshCreate;
use engine::renderer::Renderer;
use engine::resource_file::ResourceFile;
use engine::utils::thread_pool::SafeThreadPool;
use engine::utils::HashSet;
use engine::{renderer, Application, Input};
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, Vec3};
use overworld::Overworld;
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::f32::consts::FRAC_PI_2;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc};
use std::time::Instant;
use vk_wrapper::Adapter;
use winit::event::{MouseButton, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Fullscreen, Window, WindowBuilder};

const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);

pub struct Game {
    resources: Arc<ResourceFile>,
    registry: Arc<MainRegistry>,

    cursor_rel: (i32, i32),
    game_tick_finished: Arc<AtomicBool>,

    overworld: Overworld,
    overworld_streamer: Option<Arc<Mutex<OverworldStreamer>>>,

    player_pos: DVec3,
    player_aabb: AABB,
    fall_time: f64,
    curr_jump_force: f64,
    look_at_block: Option<(I64Vec3, Facing)>,
    block_set_cooldown: f64,
    curr_block: Block,

    change_stream_pos: bool,
    player_collision_enabled: bool,
    cursor_grab: bool,
}

impl Game {
    const MOVEMENT_SPEED: f64 = 8.0;
    const MOUSE_SENSITIVITY: f64 = 0.003;

    pub fn init() -> Game {
        let resources = ResourceFile::open("resources").unwrap();
        let main_registry = MainRegistry::init(&resources);
        let game_tick_finished = Arc::new(AtomicBool::new(true));
        let overworld = Overworld::new(&main_registry, 0);

        let program = Game {
            resources,
            registry: Arc::clone(&main_registry),
            cursor_rel: (0, 0),
            game_tick_finished: Arc::clone(&game_tick_finished),
            overworld,
            overworld_streamer: None,
            player_pos: Default::default(),
            player_aabb: AABB::from_size(DVec3::new(0.6, 1.75, 0.6)),
            fall_time: 0.0,
            curr_jump_force: 0.0,
            look_at_block: None,
            block_set_cooldown: 0.0,
            curr_block: main_registry.block_default(),
            change_stream_pos: true,
            player_collision_enabled: false,
            cursor_grab: true,
        };
        program
    }
}

impl Application for Game {
    fn on_engine_start(&mut self, event_loop: &EventLoop<()>) -> Window {
        let window = WindowBuilder::new()
            .with_title(PROGRAM_NAME)
            .with_inner_size(winit::dpi::PhysicalSize::new(
                DEF_WINDOW_SIZE.0,
                DEF_WINDOW_SIZE.1,
            ))
            .with_resizable(true)
            .build(event_loop)
            .unwrap();

        // Center the window
        let win_size = window.outer_size();
        let mon_size = window.current_monitor().unwrap().size();
        window.set_outer_position(winit::dpi::PhysicalPosition {
            x: (mon_size.width as i32 - win_size.width as i32) / 2,
            y: (mon_size.height as i32 - win_size.height as i32) / 2,
        });

        window.set_cursor_grab(self.cursor_grab).unwrap();
        window.set_cursor_visible(!self.cursor_grab);

        window
    }

    fn on_adapter_select(&mut self, _adapters: &[Arc<Adapter>]) -> usize {
        0
    }

    fn on_engine_initialized(&mut self, renderer: &mut Renderer) {
        renderer.set_settings(renderer::Settings {
            fps_limit: renderer::FPSLimit::VSync,
            prefer_triple_buffering: false,
            textures_mipmaps: true,
            texture_quality: renderer::TextureQuality::STANDARD,
            translucency_max_depth: renderer::TranslucencyMaxDepth::LOW,
            textures_max_anisotropy: 1.0,
        });

        let mat_pipelines;
        {
            mat_pipelines = material_pipelines::create(&self.resources, renderer);

            for (index, (ty, res_ref)) in self.registry.registry().textures().iter().enumerate() {
                renderer.load_texture_into_atlas(index as u32, *ty, res_ref.clone());
            }
            for (id, info) in self.registry.registry().materials().iter().enumerate() {
                renderer.set_material(id as u32, *info);
            }
        }

        self.player_pos = DVec3::new(0.0, 64.0, 0.0);

        let mut overworld_streamer =
            OverworldStreamer::new(&self.registry, renderer, mat_pipelines.cluster(), &self.overworld);

        // overworld_streamer.set_xz_render_distance(1024);
        overworld_streamer.set_xz_render_distance(256);
        overworld_streamer.set_y_render_distance(256);

        self.overworld_streamer = Some(Arc::new(Mutex::new(overworld_streamer)));

        // let scene = renderer.scene();
        // let tri = scene.create_entity();
        // scene.storage_write::<component::Transform>().set(
        //     tri,
        //     component::Transform::new(
        //         // DVec3::new(-0.5, -0.5, -0.5),
        //         DVec3::default(),
        //         Vec3::from_element(0.0),
        //         Vec3::from_element(1.0),
        //     ),
        // );
        // scene.storage_write::<component::RenderConfig>().set(
        //     tri,
        //     component::RenderConfig::new(renderer, mat_pipelines.cluster(), false),
        // );
        //
        // let vertices: Vec<_> = block_model::cube_quads(Vec3::from_element(0.0), Vec3::from_element(1.0))
        //     .iter()
        //     .map(|q| {
        //         q.vertices().iter().map(|v| PackedVertex {
        //             position: *v,
        //             pack0: Default::default(),
        //         })
        //     })
        //     .flatten()
        //     .collect();
        //
        // let mut indices = vec![0; vertices.len() / 4 * 6];
        // for i in (0..indices.len()).step_by(6) {
        //     let ind = (i / 6 * 4) as u32;
        //     indices[i] = ind;
        //     indices[i + 1] = ind + 2;
        //     indices[i + 2] = ind + 1;
        //     indices[i + 3] = ind + 2;
        //     indices[i + 4] = ind + 3;
        //     indices[i + 5] = ind + 1;
        // }
        //
        // scene.storage_write::<component::VertexMesh>().set(
        //     tri,
        //     component::VertexMesh::new(
        //         &renderer
        //             .device()
        //             .create_vertex_mesh(&vertices, Some(&indices[0..18]))
        //             .unwrap()
        //             .raw(),
        //     ),
        // );
    }

    fn on_update(&mut self, delta_time: f64, renderer: &mut Renderer, input: &mut Input) {
        let kb = input.keyboard();
        let mut vel_front_back = 0;
        let mut vel_left_right = 0;
        let mut vel_up_down = 0;

        if kb.is_key_pressed(VirtualKeyCode::W) {
            vel_front_back += 1;
        }
        if kb.is_key_pressed(VirtualKeyCode::S) {
            vel_front_back -= 1;
        }
        if kb.is_key_pressed(VirtualKeyCode::A) {
            vel_left_right -= 1;
        }
        if kb.is_key_pressed(VirtualKeyCode::D) {
            vel_left_right += 1;
        }
        if kb.is_key_pressed(VirtualKeyCode::Space) {
            if !self.player_collision_enabled {
                vel_up_down += 1;
            }
            // self.curr_jump_force = 5.7;
            self.curr_jump_force = 14.0;
        }
        if kb.is_key_pressed(VirtualKeyCode::LShift) {
            vel_up_down -= 1;
        }

        {
            let camera = renderer.active_camera_mut();
            let ms = Self::MOVEMENT_SPEED * delta_time;
            let mut motion_delta = DVec3::from_element(0.0);

            // Handle rotation
            let cursor_offset = (
                self.cursor_rel.0 as f32 * Self::MOUSE_SENSITIVITY as f32,
                self.cursor_rel.1 as f32 * Self::MOUSE_SENSITIVITY as f32,
            );

            let mut rotation = camera.rotation();
            rotation.x = (rotation.x + cursor_offset.1).clamp(-FRAC_PI_2, FRAC_PI_2);
            rotation.y += cursor_offset.0;
            // rotation.y += delta_time as f32;

            camera.set_rotation(rotation);
            self.cursor_rel = (0, 0);

            // Handle translation
            motion_delta.y += vel_up_down as f64 * ms;
            motion_delta += engine::utils::camera_move_xz(
                camera.rotation(),
                vel_front_back as f64 * ms,
                vel_left_right as f64 * ms,
            );

            let prev_pos = self.player_pos;

            if self.player_collision_enabled {
                {
                    let mut blocks = self.overworld.access();

                    if blocks
                        .get_block(&glm::convert_unchecked(glm::floor(&self.player_pos)))
                        .is_some()
                    {
                        // Free fall
                        motion_delta.y -= (physics::G_ACCEL * self.fall_time) * delta_time;

                        // Jump force
                        motion_delta.y += self.curr_jump_force * delta_time;
                    }
                }

                let new_pos = self
                    .overworld
                    .move_entity(prev_pos, motion_delta, &self.player_aabb);

                if new_pos.y.abs_diff_eq(&prev_pos.y, MOTION_EPSILON) {
                    // Note: the ground is reached
                    self.curr_jump_force = 0.0;
                    // println!("GROUND HIT");
                    self.fall_time = delta_time;
                } else {
                    self.fall_time += delta_time;
                }

                self.player_pos = new_pos;
            } else {
                self.player_pos = prev_pos + motion_delta;
            }

            camera.set_position(self.player_pos + DVec3::new(0.0, 0.625, 0.0));
        }

        // Set block on mouse buttons
        {
            if self.block_set_cooldown == 0.0 {
                if input.mouse().is_button_pressed(MouseButton::Left) {
                    let mut access = self.overworld.access();
                    let camera = renderer.active_camera();

                    self.look_at_block =
                        access.get_block_at_ray(&camera.position(), &glm::convert(camera.direction()), 3.0);

                    if let Some((pos, facing)) = self.look_at_block {
                        let dir: I64Vec3 = glm::convert(facing.direction());
                        let set_pos = pos + dir;

                        access.set_block(&set_pos, self.curr_block);

                        if self.curr_block == self.registry.block_glow() {
                            access.set_light(&set_pos, LightLevel::from_intensity(10));
                        } else {
                            // Remove light to cause occlusion of nearby lights
                            access.remove_light(&set_pos);
                        }

                        self.block_set_cooldown = 0.15;
                    }
                } else if input.mouse().is_button_pressed(MouseButton::Right) {
                    let mut access = self.overworld.access();
                    let camera = renderer.active_camera();

                    self.look_at_block =
                        access.get_block_at_ray(&camera.position(), &glm::convert(camera.direction()), 3.0);

                    if let Some(inter) = self.look_at_block {
                        let prev_block = access.get_block(&inter.0).unwrap().block();
                        access.set_block(&inter.0, self.registry.block_empty());

                        if prev_block == self.registry.block_glow() {
                            access.remove_light(&inter.0);
                        } else {
                            // Set corresponding light level if there is a light nearby
                            access.check_neighbour_lighting(&inter.0);
                        }

                        self.block_set_cooldown = 0.15;
                    }
                }
            }

            self.block_set_cooldown = (self.block_set_cooldown - delta_time).max(0.0);

            // if let Some(a) = blocks.set_block(I64Vec3::new(-2, 63, -5), self.registry.block_default()) {
            //     a.build();
            // }
            // if let Some(a) = a.set_block(I64Vec3::new(1, 60, 0), self.registry.block_default()) {
            //     a.build();
            // }
            // let b = a.get_block(
            //     glm::try_convert::<DVec3, I64Vec3>(self.player_pos.map(|v| v.div_euclid(1.0))).unwrap(),
            // );
            // if let Some(b) = b {
            //     println!("{} {}", b.block().archetype(), b.block().has_textured_model());
            // }
        }

        if self.game_tick_finished.swap(false, atomic::Ordering::Relaxed) {
            let streamer = Arc::clone(self.overworld_streamer.as_ref().unwrap());
            let game_tick_finished = Arc::clone(&self.game_tick_finished);

            {
                let mut streamer = streamer.lock();

                let t0 = Instant::now();
                streamer.update_renderer(renderer);
                let t1 = Instant::now();
                let el = (t1 - t0).as_secs_f64();
                if el > 0.001 {
                    println!("update_renderer time: {}", (t1 - t0).as_secs_f64());
                }

                let p = DVec3::new(self.player_pos.x, self.player_pos.y, self.player_pos.z);
                // let p = DVec3::new(43.0, 0.0, -44.0);

                if self.change_stream_pos {
                    streamer.set_stream_pos(p);
                }
            }

            coroutine_queue().spawn(|| game_tick(streamer, game_tick_finished));
            // println!("as {}", renderer.device().calc_real_device_mem_usage());
        }
    }

    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        main_window: &Window,
        control_flow: &mut ControlFlow,
    ) {
        use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};

        match event {
            Event::WindowEvent {
                window_id: _window_id,
                event,
            } => match event {
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input,
                    is_synthetic: _,
                } => {
                    if input.virtual_keycode.is_none() {
                        return;
                    }
                    if input.state == ElementState::Released {
                        match input.virtual_keycode.unwrap() {
                            VirtualKeyCode::Escape => {
                                *control_flow = ControlFlow::Exit;
                            }
                            VirtualKeyCode::L => {
                                self.change_stream_pos = !self.change_stream_pos;
                            }
                            VirtualKeyCode::C => {
                                self.player_collision_enabled = !self.player_collision_enabled;
                            }
                            VirtualKeyCode::F11 => {
                                if let Some(_) = main_window.fullscreen() {
                                    main_window.set_fullscreen(None);
                                } else {
                                    let mode = utils::find_largest_video_mode(
                                        &main_window.current_monitor().unwrap(),
                                    );
                                    main_window.set_fullscreen(Some(Fullscreen::Exclusive(mode)))
                                }
                            }
                            VirtualKeyCode::P => {
                                println!("{}", self.player_pos);
                            }
                            VirtualKeyCode::T => {
                                self.cursor_grab = !self.cursor_grab;
                                main_window.set_cursor_grab(self.cursor_grab).unwrap();
                                main_window.set_cursor_visible(!self.cursor_grab);
                            }
                            VirtualKeyCode::Key1 => {
                                self.curr_block = self.registry.block_default();
                            }
                            VirtualKeyCode::Key2 => {
                                self.curr_block = self.registry.block_glow();
                            }
                            _ => {}
                        }
                    }
                }
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                _ => {}
            },
            Event::DeviceEvent { device_id: _, event } => match event {
                DeviceEvent::MouseMotion { delta } => {
                    self.cursor_rel.0 += delta.0 as i32;
                    self.cursor_rel.1 += delta.1 as i32;
                }
                _ => {}
            },
            _ => {}
        }
    }
}

pub fn game_tick(streamer: Arc<Mutex<OverworldStreamer>>, finished: Arc<AtomicBool>) {
    let mut streamer = streamer.lock();

    // let t0 = Instant::now();
    streamer.update();
    // let t1 = Instant::now();
    // println!("tick time: {}", (t1 - t0).as_secs_f64());

    finished.store(true, atomic::Ordering::Relaxed);
}
