pub mod main_registry;
pub mod overworld;
pub mod overworld_streamer;
pub mod registry;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld::cluster::BlockDataImpl;
use crate::game::overworld_streamer::OverworldStreamer;
use crate::material_pipelines;
use crate::physics::AABB;
use engine::ecs::scene::Scene;
use engine::renderer::Renderer;
use engine::resource_file::ResourceFile;
use engine::utils::thread_pool::SafeThreadPool;
use engine::utils::HashSet;
use engine::Application;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3};
use overworld::Overworld;
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPool;
use std::f32::consts::FRAC_PI_2;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc};
use std::time::Instant;
use winit::event::{Event, VirtualKeyCode};

pub struct Game {
    resources: Arc<ResourceFile>,
    registry: Arc<MainRegistry>,

    pressed_keys: HashSet<winit::event::VirtualKeyCode>,

    cursor_rel: (i32, i32),
    game_tick_finished: Arc<AtomicBool>,

    overworld: Overworld,
    overworld_streamer: Option<Arc<Mutex<OverworldStreamer>>>,

    player_pos: DVec3,
    player_aabb: AABB,
    change_stream_pos: bool,
}

impl Game {
    const MOVEMENT_SPEED: f64 = 4.0;
    const MOUSE_SENSITIVITY: f64 = 0.003;

    pub fn init() -> Game {
        let resources = ResourceFile::open("resources").unwrap();
        let main_registry = MainRegistry::init(&resources);
        let game_tick_finished = Arc::new(AtomicBool::new(true));
        let overworld = Overworld::new(&main_registry, 0);

        let program = Game {
            resources,
            registry: main_registry,
            pressed_keys: Default::default(),
            cursor_rel: (0, 0),
            game_tick_finished: Arc::clone(&game_tick_finished),
            overworld,
            overworld_streamer: None,
            player_pos: Default::default(),
            player_aabb: AABB::from_size(DVec3::new(0.6, 1.75, 0.6)),
            change_stream_pos: true,
        };
        program
    }

    pub fn is_key_pressed(&self, keycode: winit::event::VirtualKeyCode) -> bool {
        self.pressed_keys.contains(&keycode)
    }
}

impl Application for Game {
    fn on_start(&mut self, renderer: &mut Renderer) {
        let mat_pipelines;
        {
            mat_pipelines = material_pipelines::create(&self.resources, renderer);

            for (ty, res_ref) in self.registry.registry().textures() {
                let id = renderer.add_texture(*ty, res_ref.clone());
                renderer.load_texture(id);
            }

            for (id, info) in self.registry.registry().materials().iter().enumerate() {
                renderer.set_material(id as u32, *info);
            }
        }

        renderer
            .active_camera_mut()
            .set_position(DVec3::new(0.0, 64.0, 0.0));

        let mut overworld_streamer = OverworldStreamer::new(
            &self.registry,
            renderer,
            mat_pipelines.cluster(),
            self.overworld.loaded_clusters(),
        );

        overworld_streamer.set_xz_render_distance(1024);
        overworld_streamer.set_y_render_distance(256);

        self.overworld_streamer = Some(Arc::new(Mutex::new(overworld_streamer)));
    }

    fn on_update(&mut self, delta_time: f64, renderer: &mut Renderer, background_tp: &ThreadPool) {
        let mut vel_front_back = 0;
        let mut vel_left_right = 0;
        let mut vel_up_down = 0;

        if self.is_key_pressed(VirtualKeyCode::W) {
            vel_front_back += 1;
        }
        if self.is_key_pressed(VirtualKeyCode::S) {
            vel_front_back -= 1;
        }
        if self.is_key_pressed(VirtualKeyCode::A) {
            vel_left_right -= 1;
        }
        if self.is_key_pressed(VirtualKeyCode::D) {
            vel_left_right += 1;
        }
        if self.is_key_pressed(VirtualKeyCode::Space) {
            vel_up_down += 1;
        }
        if self.is_key_pressed(VirtualKeyCode::LShift) {
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

            camera.set_rotation(rotation);

            // Handle translation
            motion_delta.y += vel_up_down as f64 * ms;
            motion_delta += engine::utils::camera_move_xz(
                camera.rotation(),
                vel_front_back as f64 * ms,
                vel_left_right as f64 * ms,
            );

            let overworld = &self.overworld;
            let new_pos = overworld.move_entity(camera.position(), motion_delta, &self.player_aabb);

            camera.set_position(new_pos);

            self.player_pos = camera.position();
            self.cursor_rel = (0, 0);
            // dbg!(self.player_pos);
        }

        // Physics
        {
            let camera = renderer.active_camera_mut();
            let overworld = &self.overworld;

            // let new_player_pos =

            // let delta = overworld.calc_collision_delta(&self.player_aabb.translate(self.player_pos));

            // println!("BEFORE DELTA: {}", camera.position());
            // camera.set_position(camera.position() - delta);
            // self.player_pos = camera.position();

            // println!("{}", delta);
            // println!("AFTER DELTA: {}", camera.position());

            let clusters = overworld.clusters();
            let mut a = clusters.access();

            if let Some(a) = a.set_block(I64Vec3::new(0, 60, 0), self.registry.block_default()) {
                a.build();
            }
            if let Some(a) = a.set_block(I64Vec3::new(1, 60, 0), self.registry.block_default()) {
                a.build();
            }

            // let b = a.get_block(
            //     glm::try_convert::<DVec3, I64Vec3>(camera.position().map(|v| v.div_euclid(1.0))).unwrap(),
            // );
            // if let Some(b) = b {
            //     println!("{} {}", b.block().archetype(), b.block().has_texture_model());
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

            background_tp.spawn(|| game_tick(streamer, game_tick_finished));
            println!("as {}", renderer.device().calc_real_device_mem_usage());
        }
    }

    fn on_event(&mut self, event: Event<()>) {
        use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};

        match event {
            Event::WindowEvent { window_id: _, event } => match event {
                WindowEvent::KeyboardInput {
                    device_id: _,
                    input,
                    is_synthetic: _,
                } => {
                    if input.state == ElementState::Pressed {
                        if let Some(keycode) = input.virtual_keycode {
                            self.pressed_keys.insert(keycode);

                            if keycode == VirtualKeyCode::L {
                                self.change_stream_pos = !self.change_stream_pos;
                            }
                        }
                    } else {
                        if let Some(keycode) = input.virtual_keycode {
                            self.pressed_keys.remove(&keycode);
                        }
                    }
                }
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

    let _t0 = Instant::now();
    streamer.update();
    let _t1 = Instant::now();
    // println!("tick time: {}", (t1 - t0).as_secs_f64());

    finished.store(true, atomic::Ordering::Relaxed);
}
