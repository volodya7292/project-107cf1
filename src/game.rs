use std::f32::consts::FRAC_PI_2;
use std::sync::atomic::AtomicBool;
use std::sync::{atomic, Arc};
use std::time::Instant;

use approx::AbsDiffEq;
use entity_data::AnyState;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec3, I64Vec3, Vec2, Vec3};
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPool;
use winit::event::{MouseButton, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowBuilder};

use core::main_registry::{MainRegistry, StatelessBlock};
use core::overworld::block::event_handlers::AfterTickActionsStorage;
use core::overworld::block::{AnyBlockState, Block, BlockState};
use core::overworld::facing::Facing;
use core::overworld::light_level::LightLevel;
use core::overworld::liquid_level::LiquidLevel;
use core::overworld::position::{BlockPos, ClusterPos};
use core::overworld::raw_cluster::{BlockDataImpl, RawCluster};
use core::overworld::Overworld;
use core::overworld::{block, block_component, raw_cluster, LoadedClusters};
use core::physics::aabb::{AABBRayIntersection, AABB};
use core::physics::MOTION_EPSILON;
use core::registry::Registry;
use engine::ecs::component;
use engine::ecs::component::simple_text::{StyledString, TextHAlign, TextStyle};
use engine::queue::{coroutine_queue, intensive_queue};
use engine::renderer::module::text_renderer::{FontSet, TextObject, TextRenderer};
use engine::renderer::Renderer;
use engine::resource_file::ResourceFile;
use engine::utils::thread_pool::SafeThreadPool;
use engine::utils::{HashMap, HashSet, MO_RELAXED};
use engine::{renderer, Application, Input};
use renderer::camera;
use vk_wrapper::Adapter;

use crate::client::overworld::overworld_streamer::OverworldStreamer;
use crate::client::{material_pipelines, utils};
use crate::default_resources::DefaultResourceMapping;
use crate::resource_mapping::ResourceMapping;
use crate::{default_resources, PROGRAM_NAME};

const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);

pub struct Game {
    resources: Arc<ResourceFile>,
    registry: Arc<MainRegistry>,
    res_map: Arc<DefaultResourceMapping>,

    cursor_rel: (f64, f64),
    game_tick_finished: Arc<AtomicBool>,

    overworld: Arc<Overworld>,
    overworld_streamer: Option<Arc<Mutex<OverworldStreamer>>>,

    player_pos: DVec3,
    player_aabb: AABB,
    fall_time: f64,
    curr_jump_force: f64,
    look_at_block: Option<(BlockPos, Facing)>,
    block_set_cooldown: f64,
    curr_block: AnyBlockState,
    set_water: bool,

    change_stream_pos: bool,
    player_collision_enabled: bool,
    cursor_grab: bool,
}

impl Game {
    const MOVEMENT_SPEED: f64 = 16.0;
    const MOUSE_SENSITIVITY: f64 = 0.2;

    pub fn init() -> Game {
        let resources = ResourceFile::open("resources").unwrap();
        let main_registry = MainRegistry::init();
        let res_map = DefaultResourceMapping::init(&main_registry, &resources);

        let game_tick_finished = Arc::new(AtomicBool::new(true));
        let overworld = Overworld::new(&main_registry, 0);

        let spawn_point = overworld.generator().gen_spawn_point();
        // dbg!(spawn_point);

        // crate::proto::make_world_prototype_image(overworld.generator());
        // crate::proto::make_climate_graph_image(main_registry.registry());

        let program = Game {
            resources,
            registry: Arc::clone(&main_registry),
            res_map,
            cursor_rel: (0.0, 0.0),
            game_tick_finished: Arc::clone(&game_tick_finished),
            overworld,
            overworld_streamer: None,
            player_pos: glm::convert(spawn_point.0),
            player_aabb: AABB::from_size(DVec3::new(0.6, 1.75, 0.6)),
            fall_time: 0.0,
            curr_jump_force: 0.0,
            look_at_block: None,
            block_set_cooldown: 0.0,
            curr_block: main_registry.block_test.into_any(),
            set_water: false,
            change_stream_pos: true,
            player_collision_enabled: false,
            cursor_grab: true,
        };
        program
    }

    pub fn grab_cursor(&mut self, window: &Window, enabled: bool) {
        self.cursor_grab = enabled;

        if self.cursor_grab {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }
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

        self.grab_cursor(&window, true);
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
            textures_max_anisotropy: 1.0,
        });

        let mat_pipelines;
        {
            mat_pipelines = material_pipelines::create(&self.resources, renderer);

            for (index, (ty, res_ref)) in self.res_map.storage().textures().iter().enumerate() {
                renderer.load_texture_into_atlas(index as u32, *ty, res_ref.clone());
            }

            for (id, material) in self.res_map.storage().materials().iter().enumerate() {
                renderer.set_material(id as u32, material.info());
            }
        }

        // self.player_pos = DVec3::new(0.5, 64.0, 0.5);

        let mut overworld_streamer = OverworldStreamer::new(
            renderer,
            mat_pipelines.cluster(),
            &self.overworld,
            Arc::clone(self.res_map.storage()),
        );

        // overworld_streamer.set_xz_render_distance(1024);
        overworld_streamer.set_xz_render_distance(256);
        overworld_streamer.set_y_render_distance(256);
        overworld_streamer.set_stream_pos(self.player_pos);

        self.overworld_streamer = Some(Arc::new(Mutex::new(overworld_streamer)));

        // -------------------------------------------------------
        let text_renderer = renderer.module_mut::<TextRenderer>().unwrap();
        let font_id = text_renderer.register_font(
            FontSet::from_bytes(
                include_bytes!("../res/fonts/Romanesco-Regular.ttf").to_vec(),
                None,
            )
            .unwrap(),
        );
        let text = renderer.add_object(TextObject::new(
            component::Transform::new(
                DVec3::new(self.player_pos.x, self.player_pos.y + 60.0, self.player_pos.z),
                Vec3::default(),
                Vec3::from_element(1.0),
            ),
            component::SimpleText::new(StyledString::new(
                "Govno, my is Gmine".to_owned(),
                TextStyle::new().with_font(font_id).with_font_size(0.5),
            ))
            .with_max_width(3.0)
            .with_h_align(TextHAlign::LEFT),
        ));
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
            let cursor_offset = Vec2::new(self.cursor_rel.0 as f32, self.cursor_rel.1 as f32)
                * (Self::MOUSE_SENSITIVITY * delta_time) as f32;

            let mut rotation = camera.rotation();
            rotation.x = (rotation.x + cursor_offset.y).clamp(-FRAC_PI_2, FRAC_PI_2);
            rotation.y += cursor_offset.x;
            // rotation.y += delta_time as f32;

            camera.set_rotation(rotation);
            self.cursor_rel = (0.0, 0.0);

            // Handle translation
            motion_delta.y += vel_up_down as f64 * ms;
            motion_delta += camera::move_xz(
                camera.rotation(),
                vel_front_back as f64 * ms,
                vel_left_right as f64 * ms,
            );

            let prev_pos = self.player_pos;

            if self.player_collision_enabled {
                {
                    let mut blocks = self.overworld.access();

                    if blocks.get_block(&BlockPos::from_f64(&self.player_pos)).is_some() {
                        // Free fall
                        motion_delta.y -= (core::physics::G_ACCEL * self.fall_time) * delta_time;

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

        // Set block on mouse button click
        {
            if self.block_set_cooldown == 0.0 {
                if input.mouse().is_button_pressed(MouseButton::Left) {
                    let mut access = self.overworld.access();
                    let camera = renderer.active_camera();

                    self.look_at_block =
                        access.get_block_at_ray(&camera.position(), &glm::convert(camera.direction()), 3.0);

                    if let Some((pos, facing)) = self.look_at_block {
                        let dir: I64Vec3 = glm::convert(facing.direction());
                        let set_pos = pos.offset(&dir);

                        if self.set_water {
                            access.set_liquid(&set_pos, self.res_map.material_water(), LiquidLevel::new(15));
                        } else {
                            access.update_block(&set_pos, |data| data.set(self.curr_block.clone()));

                            if self.curr_block.block_id == self.registry.block_glow.block_id {
                                access.set_light(&set_pos, LightLevel::from_intensity(10));
                            } else {
                                // Remove light to cause occlusion of nearby lights
                                access.remove_light(&set_pos);
                            }
                        }

                        self.block_set_cooldown = 0.15;
                    }
                } else if input.mouse().is_button_pressed(MouseButton::Right) {
                    let mut access = self.overworld.access();
                    let camera = renderer.active_camera();

                    self.look_at_block =
                        access.get_block_at_ray(&camera.position(), &glm::convert(camera.direction()), 3.0);

                    if let Some(inter) = self.look_at_block {
                        let prev_block_id = access.get_block(&inter.0).unwrap().block_id();
                        access.update_block(&inter.0, |data| data.set(self.registry.block_empty));

                        if prev_block_id == self.registry.block_glow.block_id {
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
            let overworld = Arc::clone(&self.overworld);
            let streamer = Arc::clone(self.overworld_streamer.as_ref().unwrap());
            let game_tick_finished = Arc::clone(&self.game_tick_finished);

            {
                let mut streamer = streamer.lock();

                let t0 = Instant::now();
                streamer.update_scene(renderer);
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

            coroutine_queue().spawn(|| game_tick(overworld, streamer, game_tick_finished));
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
                                println!(
                                    "{:?}",
                                    self.player_pos.map(|v| v.rem_euclid(RawCluster::SIZE as f64))
                                );
                            }
                            VirtualKeyCode::T => {
                                self.grab_cursor(main_window, !self.cursor_grab);
                                main_window.set_cursor_visible(!self.cursor_grab);
                            }
                            VirtualKeyCode::Key1 => {
                                self.curr_block = self.registry.block_test.into_any();
                                self.set_water = false;
                            }
                            VirtualKeyCode::Key2 => {
                                self.curr_block = self.registry.block_glow.into_any();
                                self.set_water = false;
                            }
                            VirtualKeyCode::Key3 => {
                                self.set_water = true;
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
                    self.cursor_rel.0 += delta.0;
                    self.cursor_rel.1 += delta.1;
                }
                _ => {}
            },
            _ => {}
        }
    }
}

pub fn game_tick(
    overworld: Arc<Overworld>,
    streamer: Arc<Mutex<OverworldStreamer>>,
    finished: Arc<AtomicBool>,
) {
    let mut streamer = streamer.lock();

    // let t0 = Instant::now();
    let dirty_clusters = streamer.update();
    // let t1 = Instant::now();
    // println!("tick time: {}", (t1 - t0).as_secs_f64());

    process_active_blocks(&overworld, &dirty_clusters);

    finished.store(true, atomic::Ordering::Relaxed);
}

pub fn process_active_blocks(overworld: &Overworld, dirty_clusters: &HashSet<ClusterPos>) {
    let loaded_clusters = overworld.loaded_clusters().read();
    let registry = overworld.main_registry().registry();

    let total_after_actions = Mutex::new(Vec::with_capacity(loaded_clusters.len()));

    loaded_clusters.par_iter().for_each(|(cl_pos, o_cluster)| {
        if o_cluster.may_have_active_blocks.load(MO_RELAXED) || dirty_clusters.contains(cl_pos) {
            let mut has_active_blocks = false;
            let cluster = o_cluster.cluster.read();

            let mut after_actions = AfterTickActionsStorage::new();

            if let Some(cluster) = &*cluster {
                for (pos, block_data) in cluster.active_blocks() {
                    let global_pos = cl_pos.to_block_pos().offset(&glm::convert(*pos.get()));
                    let block = registry.get_block(block_data.block_id()).unwrap();

                    if let Some(on_tick) = &block.event_handlers().on_tick {
                        on_tick(&global_pos, block_data, overworld, after_actions.builder());
                        has_active_blocks = true;
                    }
                }
            }
            o_cluster
                .may_have_active_blocks
                .store(has_active_blocks, MO_RELAXED);

            total_after_actions.lock().push(after_actions);
        }
    });

    let mut cluster_accessor = overworld.access();
    let total_after_actions = total_after_actions.lock();

    // Note: first set components, and only after that set states which may be of different archetypes.
    for actions in &*total_after_actions {
        for info in &actions.components_infos {
            let success = (info.apply_fn)(&mut cluster_accessor, &info.pos, info.ptr);
        }
    }

    for actions in &*total_after_actions {
        for info in &actions.states_infos {
            let success = (info.apply_fn)(&mut cluster_accessor, &info.pos, info.ptr);
        }
    }
}
