mod main_registry;
mod overworld;
pub mod overworld_streamer;
pub mod registry;

use crate::game::main_registry::MainRegistry;
use crate::game::overworld_streamer::OverworldStreamer;
use crate::renderer;
use crate::renderer::material_pipelines::MaterialPipelines;
use crate::renderer::{component, Renderer};
use crate::utils::{HashMap, HashSet};
use entity_data::EntityStorageLayout;
use futures::FutureExt;
use nalgebra as na;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, Vec3};
use overworld::Overworld;
use registry::Registry;
use simdnoise::NoiseBuilder;
use std::f32::consts::FRAC_PI_2;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc, Mutex};
use std::thread;
use std::time::Instant;
use winit::event::VirtualKeyCode;

pub struct Game {
    pub(crate) renderer: Arc<Mutex<Renderer>>,

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

    pub fn on_update(&mut self, delta_time: f64) {
        use winit::event::VirtualKeyCode;

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
            let mut camera_comps = camera_comps.write().unwrap();
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

            // dbg!(camera.position());
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
                    // println!("update_renderer time: {}", (t1 - t0).as_secs_f64());

                    let p = DVec3::new(self.player_pos.x, 0.0, self.player_pos.z);
                    // let p = DVec3::new(220.0, 0.0, -22.0);

                    if self.change_stream_pos {
                        streamer.set_stream_pos(p);
                    }
                }

                rayon::spawn(|| game_tick(streamer, overworld, game_tick_finished));
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

pub fn new(renderer: &Arc<Mutex<Renderer>>, mat_pipelines: &MaterialPipelines) -> Game {
    // renderer.lock().unwrap().set_material(
    //     0,
    //     renderer::MaterialInfo {
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
        renderer::MaterialInfo {
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
        renderer::MaterialInfo {
            diffuse_tex_id: u32::MAX,
            specular_tex_id: u32::MAX,
            normal_tex_id: u32::MAX,
            _pad: 0,
            diffuse: na::Vector4::new(1.0, 1.0, 0.0, 1.0),
            specular: Default::default(),
            emission: Default::default(),
        },
    );

    let main_registry = Arc::new(MainRegistry::init());
    let game_tick_finished = Arc::new(AtomicBool::new(true));
    let mut overworld = Overworld::new(&main_registry, 0);
    let mut overworld_streamer = overworld_streamer::new(&main_registry, renderer, &mat_pipelines.cluster());

    overworld_streamer.set_render_distance(128);
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

    /*let device = renderer.lock().unwrap().device().clone();
    let mut cluster = cluster::new(&device, 1);
    let mut cluster2 = cluster::new(&device, 1);
    let mut cluster3 = cluster::new(&device, 1);
    let mut cluster4 = cluster::new(&device, 1);

    {
        let noise =
            NoiseBuilder::gradient_3d_offset(0.0, cluster::SIZE, 0.0, cluster::SIZE, 0.0, cluster::SIZE)
                .with_seed(0)
                //.with_freq(1.0 / 20.0)
                .generate();

        let sample_noise = |x, y, z| -> f32 {
            // 18x16x16
            //noise.0[z * (cluster::SIZE * 2) * cluster::SIZE + y * (cluster::SIZE * 2) + x] * 35.0

            let n = noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0;

            ((n as f32 + (63 - y) as f32 / 10.0) / 2.0).max(0.0).min(1.0)
        };

        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..(cluster::SIZE) {
            for y in 0..(cluster::SIZE) {
                for z in 0..(cluster::SIZE) {
                    let n_v = sample_noise(x, y, z);

                    //let n_v = ((x as f32) / (cluster::SIZE as f32)).min(1.0);

                    /*let v = (na::Vector3::new(
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                    ) - na::Vector3::new(x as f32, y as f32, z as f32))
                    .magnitude()
                        / (cluster::SIZE as f32)
                        * 1.05;*/

                    points.push(cluster::DensityPointInfo {
                        pos: [x as u8, y as u8, z as u8, 0],
                        point: cluster::DensityPoint {
                            //density: (((cluster::SIZE - y - 1) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: ((x as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            density: (((64 - y) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: (n_v * 255.0) as u8,
                            //density: 255 - (v * 255.0) as u8,
                            material: 0,
                        },
                    });
                }
            }
        }

        cluster.set_densities(&points);
    }

    {
        let noise =
            NoiseBuilder::gradient_3d_offset(64.0, cluster::SIZE, 0.0, cluster::SIZE, 0.0, cluster::SIZE)
                .with_seed(0)
                //.with_freq(1.0 / 20.0)
                .generate();

        let sample_noise = |x, y, z| -> f32 {
            // 18x16x16
            //noise.0[z * (cluster::SIZE * 2) * cluster::SIZE + y * (cluster::SIZE * 2) + x] * 35.0

            let n = noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0;

            ((n as f32 + (63 - y) as f32 / 10.0) / 2.0).max(0.0).min(1.0)
        };

        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..cluster::SIZE {
            for y in 0..cluster::SIZE {
                for z in 0..cluster::SIZE {
                    let n_v = sample_noise(x, y, z);

                    /*let v = (na::Vector3::new(
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                    ) - na::Vector3::new(x as f32, y as f32, z as f32))
                    .magnitude()
                        / (cluster::SIZE as f32)
                        * 1.05;*/

                    points.push(cluster::DensityPointInfo {
                        pos: [x as u8, y as u8, z as u8, 0],
                        point: cluster::DensityPoint {
                            //density: (((cluster::SIZE - y - 1) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: ((x as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            density: (((64 - y) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: (n_v * 255.0) as u8,
                            //density: 255 - (v * 255.0) as u8,
                            material: 0,
                        },
                    });
                }
            }
        }

        cluster2.set_densities(&points);
    }

    {
        let noise =
            NoiseBuilder::gradient_3d_offset(0.0, cluster::SIZE, 0.0, cluster::SIZE, 64.0, cluster::SIZE)
                .with_seed(0)
                //.with_freq(1.0 / 20.0)
                .generate();

        let sample_noise = |x, y, z| -> f32 {
            // 18x16x16
            //noise.0[z * (cluster::SIZE * 2) * cluster::SIZE + y * (cluster::SIZE * 2) + x] * 35.0

            let n = noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0;

            ((n as f32 + (63 - y) as f32 / 10.0) / 2.0).max(0.0).min(1.0)
        };

        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..cluster::SIZE {
            for y in 0..cluster::SIZE {
                for z in 0..cluster::SIZE {
                    let n_v = sample_noise(x, y, z);

                    /*let v = (na::Vector3::new(
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                    ) - na::Vector3::new(x as f32, y as f32, z as f32))
                    .magnitude()
                        / (cluster::SIZE as f32)
                        * 1.05;*/

                    points.push(cluster::DensityPointInfo {
                        pos: [x as u8, y as u8, z as u8, 0],
                        point: cluster::DensityPoint {
                            //density: (((cluster::SIZE - y - 1) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: ((x as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            density: (((64 - y) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: (n_v * 255.0) as u8,
                            //density: 255 - (v * 255.0) as u8,
                            material: 0,
                        },
                    });
                }
            }
        }

        cluster3.set_densities(&points);
    }

    {
        let noise =
            NoiseBuilder::gradient_3d_offset(64.0, cluster::SIZE, 0.0, cluster::SIZE, 64.0, cluster::SIZE)
                .with_seed(0)
                //.with_freq(1.0 / 20.0)
                .generate();

        let sample_noise = |x, y, z| -> f32 {
            // 18x16x16
            //noise.0[z * (cluster::SIZE * 2) * cluster::SIZE + y * (cluster::SIZE * 2) + x] * 35.0

            let n = noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0;

            ((n as f32 + (63 - y) as f32 / 10.0) / 2.0).max(0.0).min(1.0)
        };

        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..cluster::SIZE {
            for y in 0..cluster::SIZE {
                for z in 0..cluster::SIZE {
                    let n_v = sample_noise(x, y, z);

                    /*let v = (na::Vector3::new(
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                    ) - na::Vector3::new(x as f32, y as f32, z as f32))
                    .magnitude()
                        / (cluster::SIZE as f32)
                        * 1.05;*/

                    points.push(cluster::DensityPointInfo {
                        pos: [x as u8, y as u8, z as u8, 0],
                        point: cluster::DensityPoint {
                            //density: (((cluster::SIZE - y - 1) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            density: (((64 - y) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: (n_v * 255.0) as u8,
                            //density: 255 - (v * 255.0) as u8,
                            material: 0,
                        },
                    });
                }
            }
        }

        cluster4.set_densities(&points);
    }

    {
        let mut seam = cluster::Seam::new(cluster.node_size());
        let mut seam2 = cluster::Seam::new(cluster2.node_size());
        let mut seam3 = cluster::Seam::new(cluster3.node_size());
        let mut seam4 = cluster::Seam::new(cluster4.node_size());

        seam2.insert(&mut cluster4, na::Vector3::new(0, 0, 64));
        seam3.insert(&mut cluster4, na::Vector3::new(64, 0, 0));

        cluster2.fill_seam_densities(&seam2);
        cluster3.fill_seam_densities(&seam3);

        seam.insert(&mut cluster2, na::Vector3::new(64, 0, 0));
        seam.insert(&mut cluster3, na::Vector3::new(0, 0, 64));
        seam.insert(&mut cluster4, na::Vector3::new(64, 0, 64));

        cluster.fill_seam_densities(&seam);

        // -----------------------------------------------------------------------------------

        let t0 = Instant::now();
        cluster.update_mesh(&seam, 1.0); // TODO: include neighbour nodes
        let t1 = Instant::now();

        println!("CL TIME: {}", t1.duration_since(t0).as_secs_f64());

        let t0 = Instant::now();
        cluster2.update_mesh(&seam2, 1.0);
        let t1 = Instant::now();

        cluster3.update_mesh(&seam3, 1.0);
        cluster4.update_mesh(&seam4, 1.0);

        println!("CL TIME: {}", t1.duration_since(t0).as_secs_f64());
    }

    {
        let mut renderer = program.renderer.lock().unwrap();

        renderer
            .world_mut()
            .create_entity()
            .with(component::Transform::new(
                na::Vector3::new(0.0, -(cluster::SIZE as f32) / 2.0 - 4.0, 0.0),
                na::Vector3::new(0.0, 0.0, 0.0),
                na::Vector3::new(1.0, 1.0, 1.0),
            ))
            .with(component::VertexMeshRef::new(&cluster.vertex_mesh().raw()))
            .with(component::Renderer::new(&device, &mat_pipelines.cluster(), false))
            .build();
        renderer
            .world_mut()
            .create_entity()
            .with(component::Transform::new(
                na::Vector3::new(64.0, -(cluster::SIZE as f32) / 2.0 - 4.0, 0.0),
                na::Vector3::new(0.0, 0.0, 0.0),
                na::Vector3::new(1.0, 1.0, 1.0),
            ))
            .with(component::VertexMeshRef::new(&cluster2.vertex_mesh().raw()))
            .with(component::Renderer::new(&device, &mat_pipelines.cluster(), false))
            .build();
        renderer
            .world_mut()
            .create_entity()
            .with(component::Transform::new(
                na::Vector3::new(0.0, -(cluster::SIZE as f32) / 2.0 - 4.0, 64.0),
                na::Vector3::new(0.0, 0.0, 0.0),
                na::Vector3::new(1.0, 1.0, 1.0),
            ))
            .with(component::VertexMeshRef::new(&cluster3.vertex_mesh().raw()))
            .with(component::Renderer::new(&device, &mat_pipelines.cluster(), false))
            .build();
        renderer
            .world_mut()
            .create_entity()
            .with(component::Transform::new(
                na::Vector3::new(64.0, -(cluster::SIZE as f32) / 2.0 - 4.0, 64.0),
                na::Vector3::new(0.0, 0.0, 0.0),
                na::Vector3::new(1.0, 1.0, 1.0),
            ))
            .with(component::VertexMeshRef::new(&cluster4.vertex_mesh().raw()))
            .with(component::Renderer::new(&device, &mat_pipelines.cluster(), false))
            .build();
    }*/

    /*let device = renderer.lock().unwrap().device().clone();
    let mut cluster = cluster::new(&device, 1);
    let mut cluster2 = cluster::new(&device, 2);

    {
        let noise =
            NoiseBuilder::gradient_3d_offset(0.0, cluster::SIZE, 0.0, cluster::SIZE, 0.0, cluster::SIZE)
                .with_seed(0)
                .with_freq(1.0 / 50.0)
                .generate();

        let sample_noise = |x, y, z| -> f32 {
            noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0
        };
        let sample_f = |x: usize, y: usize, z: usize| -> f32 {
            let n = sample_noise(x.min(63), y.min(63), z.min(63));
            /*let f: f32 = ((544.0_f32 - y as f32) / (1024.0_f32) + n / 32.0_f32)
            .max(0.0)
            .min(1.0);*/
            let f: f32 = ((8224.0 - y as f32) / (16384.0) + n / 512.0).max(0.0).min(1.0);
            f
        };

        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..(cluster::SIZE) {
            for y in 0..(cluster::SIZE) {
                for z in 0..(cluster::SIZE) {
                    //let n_v = ((x as f32) / (cluster::SIZE as f32)).min(1.0);

                    /*let v = (na::Vector3::new(
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                    ) - na::Vector3::new(x as f32, y as f32, z as f32))
                    .magnitude()
                        / (cluster::SIZE as f32)
                        * 1.05;*/

                    //let n_v = ((n_v as f32 + (64 - (y as i32)) as f32 / 10.0) / 2.0)
                    //    .max(0.0)
                    //    .min(1.0);

                    let v = sample_f(x, y, z);
                    let v_z0 = sample_f(x, y, z.saturating_sub(1));
                    let v_z1 = sample_f(x, y, z + 1);
                    let v_y0 = sample_f(x, y.saturating_sub(1), z);
                    let v_y1 = sample_f(x, y + 1, z);
                    let v_x0 = sample_f(x.saturating_sub(1), y, z);
                    let v_x1 = sample_f(x + 1, y, z);

                    let mut n_v = v;

                    // 0.5 - ISO VALUE

                    let ad = v - 0.5;
                    let bd = v_y1 - 0.5;
                    let cd = v_x1 - 0.5;
                    let ed = v_z1 - 0.5;
                    let bd2 = v_y0 - 0.5;
                    let cd2 = v_x0 - 0.5;
                    let ed2 = v_z0 - 0.5;
                    let ta = ((v >= 0.5) as i8 as f32 - v) / ad;
                    let tb = ((v_y1 >= 0.5) as i8 as f32 - v_y1) / bd;
                    let tc = ((v_x1 >= 0.5) as i8 as f32 - v_x1) / cd;
                    let te = ((v_z1 >= 0.5) as i8 as f32 - v_z1) / ed;
                    let tb2 = ((v_y0 >= 0.5) as i8 as f32 - v_y0) / bd2;
                    let tc2 = ((v_x0 >= 0.5) as i8 as f32 - v_x0) / cd2;
                    let te2 = ((v_z0 >= 0.5) as i8 as f32 - v_z0) / ed2;
                    let t = ta.min(tb).min(tc).min(te).min(tb2).min(tc2).min(te2);
                    n_v = n_v + ad * t;

                    points.push(cluster::DensityPointInfo {
                        pos: [x as u8, y as u8, z as u8, 0],
                        point: cluster::DensityPoint {
                            //density: (((cluster::SIZE - y - 1) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: ((x as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: (((64 - y) as f32 / cluster::SIZE as f32) * 255.0) as u8 ,
                            density: (n_v * 255.0) as u8,
                            //density: 255 - (v * 255.0) as u8,
                            material: 0,
                        },
                    });
                }
            }
        }

        cluster.set_densities(&points);
    }

    {
        let noise = NoiseBuilder::gradient_3d_offset(
            64.0 / 2.0,
            cluster::SIZE,
            0.0,
            cluster::SIZE,
            0.0,
            cluster::SIZE,
        )
        .with_seed(0)
        .with_freq(1.0 / 25.0)
        .generate();

        let sample_noise = |x, y, z| -> f32 {
            noise.0[z * (cluster::SIZE) * (cluster::SIZE) + y * (cluster::SIZE) + x] * 35.0
        };

        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..cluster::SIZE {
            for y in 0..cluster::SIZE {
                for z in 0..cluster::SIZE {
                    let n_v = sample_noise(x, y, z);

                    /*let v = (na::Vector3::new(
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                        cluster::SIZE as f32 / 2.0,
                    ) - na::Vector3::new(x as f32, y as f32, z as f32))
                    .magnitude()
                        / (cluster::SIZE as f32)
                        * 1.05;*/

                    let n_v = ((n_v as f32 + (64 - (y as i32) * 2) as f32 / 10.0) / 2.0)
                        .max(0.0)
                        .min(1.0);

                    points.push(cluster::DensityPointInfo {
                        pos: [x as u8, y as u8, z as u8, 0],
                        point: cluster::DensityPoint {
                            //density: (((cluster::SIZE - y - 1) as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: ((x as f32 / cluster::SIZE as f32) * 255.0) as u8,
                            //density: (((64 - y) as f32 / (cluster::SIZE as f32 * 1.5)) * 255.0) as u8,
                            density: (n_v * 255.0) as u8,
                            //density: 255 - (v * 255.0) as u8,
                            material: 0,
                        },
                    });
                }
            }
        }

        cluster2.set_densities(&points);
    }

    {
        let mut seam = cluster::Seam::new(cluster.node_size());
        let mut seamt = cluster::Seam::new(cluster.node_size());
        let mut seam2 = cluster::Seam::new(cluster2.node_size());

        seam.insert(&mut cluster2, na::Vector3::new(64, 0, 0));

        cluster.fill_seam_densities(&seam);

        // -----------------------------------------------------------------------------------

        let t0 = Instant::now();
        cluster.update_mesh(&seam, 1.0); // TODO: include neighbour nodes
        let t1 = Instant::now();

        println!("CL TIME: {}", t1.duration_since(t0).as_secs_f64());

        let t0 = Instant::now();
        cluster2.update_mesh(&seam2, 1.0);
        let t1 = Instant::now();

        println!("CL TIME: {}", t1.duration_since(t0).as_secs_f64());
    }

    {
        let mut renderer = program.renderer.lock().unwrap();

        renderer
            .world_mut()
            .create_entity()
            .with(component::Transform::new(
                na::Vector3::new(0.0, -(cluster::SIZE as f32) / 2.0 - 4.0, 0.0),
                na::Vector3::new(0.0, 0.0, 0.0),
                na::Vector3::new(1.0, 1.0, 1.0),
            ))
            .with(component::VertexMeshRef::new(&cluster.vertex_mesh().raw()))
            .with(component::Renderer::new(&device, &mat_pipelines.cluster(), false))
            .build();
        renderer
            .world_mut()
            .create_entity()
            .with(component::Transform::new(
                na::Vector3::new(64.0, -(cluster::SIZE as f32) / 2.0 - 4.0, 0.0),
                na::Vector3::new(0.0, 0.0, 0.0),
                na::Vector3::new(1.0, 1.0, 1.0),
            ))
            .with(component::VertexMeshRef::new(&cluster2.vertex_mesh().raw()))
            .with(component::Renderer::new(&device, &mat_pipelines.cluster(), false))
            .build();
    }*/

    program
}
