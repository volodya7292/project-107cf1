use crate::object::cluster;
use crate::renderer::material_pipelines::MaterialPipelines;
use crate::renderer::{component, Renderer};
use crate::world;
use dual_contouring as dc;
use nalgebra as na;
use simdnoise::NoiseBuilder;
use specs::{Builder, WorldExt};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use vk_wrapper as vkw;

pub struct Program {
    pub(crate) renderer: Arc<Mutex<Renderer>>,

    pressed_keys: HashSet<sdl2::keyboard::Scancode>,

    cursor_rel: (i32, i32),
}

impl Program {
    const MOVEMENT_SPEED: f32 = 32.0;
    const MOUSE_SENSITIVITY: f32 = 0.005;

    pub fn init(&self) {}

    pub fn on_event(&mut self, event: sdl2::event::Event) {
        use sdl2::event::Event;
        match event {
            Event::KeyDown {
                timestamp: _,
                window_id: _,
                keycode: _,
                scancode,
                keymod: _,
                repeat: _,
            } => {
                if let Some(scancode) = scancode {
                    self.pressed_keys.insert(scancode);
                }
            }
            Event::KeyUp {
                timestamp: _,
                window_id: _,
                keycode: _,
                scancode,
                keymod: _,
                repeat: _,
            } => {
                if let Some(scancode) = scancode {
                    self.pressed_keys.remove(&scancode);
                }
            }
            Event::MouseMotion {
                timestamp: _,
                window_id: _,
                which: _,
                mousestate: _,
                x: _,
                y: _,
                xrel,
                yrel,
            } => {
                self.cursor_rel = (xrel, yrel);
            }
            _ => {}
        }
    }

    pub fn is_key_pressed(&self, scancode: sdl2::keyboard::Scancode) -> bool {
        self.pressed_keys.contains(&scancode)
    }

    pub fn on_update(&mut self, delta_time: f64) {
        {
            use sdl2::keyboard::Scancode;

            let mut vel_front_back = 0;
            let mut vel_left_right = 0;
            let mut vel_up_down = 0;

            if self.is_key_pressed(Scancode::W) {
                vel_front_back += 1;
            }
            if self.is_key_pressed(Scancode::S) {
                vel_front_back -= 1;
            }
            if self.is_key_pressed(Scancode::A) {
                vel_left_right -= 1;
            }
            if self.is_key_pressed(Scancode::D) {
                vel_left_right += 1;
            }
            if self.is_key_pressed(Scancode::Space) {
                vel_up_down += 1;
            }
            if self.is_key_pressed(Scancode::LShift) {
                vel_up_down -= 1;
            }

            let renderer = self.renderer.lock().unwrap();
            let entity = renderer.get_active_camera();
            let mut camera_comp = renderer.world().write_component::<component::Camera>();
            let camera = camera_comp.get_mut(entity).unwrap();

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

            rotation.x = na::clamp(
                rotation.x + cursor_offset.1,
                -std::f32::consts::FRAC_PI_2,
                std::f32::consts::FRAC_PI_2,
            );
            rotation.y += cursor_offset.0;

            camera.set_rotation(rotation);

            self.cursor_rel = (0, 0);

            //dbg!(camera.position());
        }

        {
            /*let renderer = self.renderer.lock().unwrap();
            let world = renderer.world();
            let cluster_comp = world.read_component::<cluster::Cluster>();

            for cluster {}

            for */
        }
    }
}

pub fn new(renderer: &Arc<Mutex<Renderer>>, mat_pipelines: &MaterialPipelines) -> Program {
    let program = Program {
        renderer: Arc::clone(renderer),
        pressed_keys: Default::default(),
        cursor_rel: (0, 0),
    };

    /*let mut world_streamer = world::streamer::new(renderer, &mat_pipelines.cluster());
    world_streamer.set_render_distance(128);
    world_streamer.set_stream_pos(na::Vector3::new(32.0, 32.0, 32.0));
    world_streamer.on_update();*/

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

        seam2.insert(&mut cluster4, na::Vector3::new(0, 0, 1));
        seam3.insert(&mut cluster4, na::Vector3::new(1, 0, 0));

        cluster2.fill_seam_densities(&seam2);
        cluster3.fill_seam_densities(&seam3);

        seam.insert(&mut cluster2, na::Vector3::new(1, 0, 0));
        seam.insert(&mut cluster3, na::Vector3::new(0, 0, 1));
        seam.insert(&mut cluster4, na::Vector3::new(1, 0, 1));

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

    let device = renderer.lock().unwrap().device().clone();
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

                    if v >= 0.5
                        && v_z0 >= 0.5
                        && v_z1 >= 0.5
                        && v_y0 >= 0.5
                        && v_y1 >= 0.5
                        && v_x0 >= 0.5
                        && v_x1 >= 0.5
                    {
                        n_v = 1.0;
                    } else if v < 0.5
                        && v_z0 < 0.5
                        && v_z1 < 0.5
                        && v_y0 < 0.5
                        && v_y1 < 0.5
                        && v_x0 < 0.5
                        && v_x1 < 0.5
                    {
                        n_v = 0.0;
                    } else {
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
                    }
                    /*

                    h: 320

                    64 * 5 = 320;
                    f: y / 320
                    div: 0.0031

                    64
                    f: (y - 128) / 64
                    div: 0.015

                     */

                    /*


                       float n = perlin_fbm(uv * 5.0, 1.0, 3) + (uv.y + 0.0) * 10.0;

                       return vec4(vec3(n), 1);

                       if (n > uv.y) {
                           return vec4(1);
                       } else {
                           return vec4(0);
                       }


                    */

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
    }

    program
}
