use crate::object::cluster;
use crate::renderer::material_pipelines::MaterialPipelines;
use crate::renderer::{component, Renderer};
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
    const MOVEMENT_SPEED: f32 = 8.0;
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

pub fn new(
    renderer: &Arc<Mutex<Renderer>>,
    device: &Arc<vkw::Device>,
    mat_pipelines: &MaterialPipelines,
) -> Program {
    let program = Program {
        renderer: Arc::clone(renderer),
        pressed_keys: Default::default(),
        cursor_rel: (0, 0),
    };

    let mut cluster = cluster::new(device);
    let mut cluster2 = cluster::new(device);

    let noise = NoiseBuilder::gradient_3d_offset(
        0.0,
        cluster::SIZE * 2,
        0.0,
        cluster::SIZE * 2,
        0.0,
        cluster::SIZE * 2,
    )
    .with_seed(0)
    //.with_freq(20.0)
    .generate();

    let sample_noise = |x, y, z| -> f32 {
        // 18x16x16
        //noise.0[z * (cluster::SIZE * 2) * cluster::SIZE + y * (cluster::SIZE * 2) + x] * 35.0

        noise.0[z * (cluster::SIZE * 2) * (cluster::SIZE * 2) + y * (cluster::SIZE * 2) + x] * 35.0
    };

    {
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
        let mut points = Vec::<cluster::DensityPointInfo>::new();

        for x in 0..cluster::SIZE {
            for y in 0..cluster::SIZE {
                for z in 0..cluster::SIZE {
                    let n_v = sample_noise(64 + x, y, z);

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
        let nei = cluster2.collect_nodes_for_seams(0);
        let mut nei_filtered = Vec::with_capacity(nei.len());

        for node in &nei {
            let pos = node.position();

            if pos.x == 0 {
                let new_pos = na::Vector3::new(cluster::SIZE as u32 * cluster2.node_size(), pos.y, pos.z);
                let mut new_data = *node.data();

                if let Some(vertex_pos) = &mut new_data.vertex_pos {
                    vertex_pos.x += cluster::SIZE as f32 * cluster2.node_size() as f32;
                }

                nei_filtered.push(dc::octree::LeafNode::new(new_pos, node.size(), new_data));
            }
        }

        let t0 = Instant::now();
        cluster.update_mesh(&nei_filtered, 1.0); // TODO: include neighbour nodes
        let t1 = Instant::now();

        println!("CL TIME: {}", t1.duration_since(t0).as_secs_f64());

        let t0 = Instant::now();
        cluster2.update_mesh(&[], 1.0);
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
            .with(component::Renderer::new(device, &mat_pipelines.cluster(), false))
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
            .with(component::Renderer::new(device, &mat_pipelines.cluster(), false))
            .build();
    }

    program
}
