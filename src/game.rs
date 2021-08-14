mod main_registry;
mod overworld;
pub mod overworld_streamer;
pub mod registry;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld_streamer::OverworldStreamer;
use crate::render_engine;
use crate::render_engine::material_pipelines::MaterialPipelines;
use crate::render_engine::{component, RenderEngine};
use crate::utils::{HashMap, HashSet};
use entity_data::EntityStorageLayout;
use futures::FutureExt;
use nalgebra as na;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, Vec3};
use overworld::Overworld;
use rayon::prelude::*;
use rayon::ThreadPool;
use registry::Registry;
use simdnoise::NoiseBuilder;
use std::f32::consts::FRAC_PI_2;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc, Mutex};
use std::thread;
use std::thread::Thread;
use std::time::Instant;
use winit::event::VirtualKeyCode;

pub struct Game {
    renderer: Arc<Mutex<RenderEngine>>,

    pressed_keys: HashSet<winit::event::VirtualKeyCode>,

    cursor_rel: (i32, i32),
    game_tick_finished: Arc<AtomicBool>,

    loaded_overworld: Option<Arc<Mutex<Overworld>>>,
    overworld_streamer: Arc<Mutex<OverworldStreamer>>,

    player_pos: DVec3,
    change_stream_pos: bool,
}

impl Game {
    const MOVEMENT_SPEED: f32 = 32.0;
    const MOUSE_SENSITIVITY: f32 = 0.003;

    pub fn on_event(&mut self, event: winit::event::Event<()>) {
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

    pub fn is_key_pressed(&self, keycode: winit::event::VirtualKeyCode) -> bool {
        self.pressed_keys.contains(&keycode)
    }

    pub fn on_update(&mut self, delta_time: f64, thread_pool: &ThreadPool) {
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
            let renderer = self.renderer.lock().unwrap();
            let entity = renderer.get_active_camera();
            let camera_comps = renderer.scene().storage::<component::Camera>();
            let mut camera_comps = camera_comps.write();
            let camera = camera_comps.get_mut(entity).unwrap();

            let ms = Self::MOVEMENT_SPEED * delta_time as f32;

            let mut pos = camera.position();
            pos.y += vel_up_down as f32 * ms;

            camera.set_position(pos);
            camera.move2(vel_front_back as f32 * ms, vel_left_right as f32 * ms);

            let mut rotation = camera.rotation();
            let cursor_offset = (
                self.cursor_rel.0 as f32 * Self::MOUSE_SENSITIVITY,
                self.cursor_rel.1 as f32 * Self::MOUSE_SENSITIVITY,
            );

            rotation.x = (rotation.x + cursor_offset.1).clamp(-FRAC_PI_2, FRAC_PI_2);
            rotation.y += cursor_offset.0;

            camera.set_rotation(rotation);

            self.player_pos = glm::convert(pos);
            self.cursor_rel = (0, 0);
            // dbg!(self.player_pos);
        }

        if self.game_tick_finished.swap(false, atomic::Ordering::Relaxed) {
            if let Some(overworld) = self.loaded_overworld.clone() {
                let streamer = Arc::clone(&self.overworld_streamer);
                let game_tick_finished = Arc::clone(&self.game_tick_finished);

                {
                    let mut streamer = streamer.lock().unwrap();
                    let t0 = Instant::now();
                    streamer.update_renderer(&mut overworld.lock().unwrap());
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

                thread_pool.spawn(|| game_tick(streamer, overworld, game_tick_finished));
                println!(
                    "as {}",
                    self.renderer
                        .lock()
                        .unwrap()
                        .device()
                        .calc_real_device_mem_usage()
                );
            }
        }
    }
}

pub fn game_tick(
    streamer: Arc<Mutex<OverworldStreamer>>,
    overworld: Arc<Mutex<Overworld>>,
    finished: Arc<AtomicBool>,
) {
    let mut streamer = streamer.lock().unwrap();
    let mut overworld = overworld.lock().unwrap();

    let t0 = Instant::now();
    streamer.update(&mut overworld);
    let t1 = Instant::now();
    // println!("tick time: {}", (t1 - t0).as_secs_f64());

    finished.store(true, atomic::Ordering::Relaxed);
}

pub fn new(renderer: &Arc<Mutex<RenderEngine>>, mat_pipelines: &MaterialPipelines) -> Game {
    // render_engine.lock().unwrap().set_material(
    //     0,
    //     render_engine::MaterialInfo {
    //         diffuse_tex_id: u32::MAX,
    //         specular_tex_id: u32::MAX,
    //         normal_tex_id: u32::MAX,
    //         _pad: 0,
    //         diffuse: na::Vector4::new(0.7, 0.3, 0.5, 1.0),
    //         specular: Default::default(),
    //         emission: Default::default(),
    //     },
    // );
    renderer.lock().unwrap().set_material(
        1,
        render_engine::MaterialInfo {
            diffuse_tex_id: u32::MAX,
            specular_tex_id: u32::MAX,
            normal_tex_id: u32::MAX,
            _pad: 0,
            diffuse: na::Vector4::new(0.3, 0.5, 0.9, 1.0),
            specular: Default::default(),
            emission: Default::default(),
        },
    );
    renderer.lock().unwrap().set_material(
        2,
        render_engine::MaterialInfo {
            diffuse_tex_id: u32::MAX,
            specular_tex_id: u32::MAX,
            normal_tex_id: u32::MAX,
            _pad: 0,
            diffuse: na::Vector4::new(1.0, 1.0, 0.0, 1.0),
            specular: Default::default(),
            emission: Default::default(),
        },
    );

    let main_registry = MainRegistry::init();
    let game_tick_finished = Arc::new(AtomicBool::new(true));
    let mut overworld = Overworld::new(&main_registry, 0);
    let mut overworld_streamer = overworld_streamer::new(&main_registry, renderer, mat_pipelines.cluster());

    overworld_streamer.set_render_distance(512);
    // overworld_streamer.update(&mut overworld);
    // overworld_streamer.update_renderer(&mut overworld);

    let program = Game {
        renderer: Arc::clone(renderer),
        pressed_keys: Default::default(),
        cursor_rel: (0, 0),
        game_tick_finished: Arc::clone(&game_tick_finished),
        loaded_overworld: Some(Arc::new(Mutex::new(overworld))),
        overworld_streamer: Arc::new(Mutex::new(overworld_streamer)),
        player_pos: Default::default(),
        change_stream_pos: true,
    };
    program
}
