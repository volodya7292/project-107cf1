mod ui;

use crate::client::utils;
use crate::default_resources;
use crate::default_resources::DefaultResourceMapping;
use crate::rendering::material_pipelines;
use crate::rendering::overworld_renderer::OverworldRenderer;
use crate::rendering::ui::fancy_button::{FancyButton, FancyButtonAccess, FancyButtonImpl};
use crate::rendering::ui::image::{ImageAccess, ImageFitness, ImageImpl, ImageSource, UIImage};
use crate::rendering::ui::text::{TextAccess, UIText, UITextImpl};
use crate::rendering::ui::{fancy_button, register_ui_elements, text, UIContext};
use crate::resource_mapping::ResourceMapping;
use approx::AbsDiffEq;
use base::execution::default_queue;
use base::execution::timer::IntervalTimer;
use base::execution::virtual_processor::VirtualProcessor;
use base::main_registry::MainRegistry;
use base::overworld::accessor::ClustersAccessorCache;
use base::overworld::accessor::ReadOnlyOverworldAccessor;
use base::overworld::accessor::ReadOnlyOverworldAccessorImpl;
use base::overworld::actions_storage::OverworldActionsStorage;
use base::overworld::actions_storage::StateChangeInfo;
use base::overworld::block::{AnyBlockState, Block, BlockState};
use base::overworld::facing::Facing;
use base::overworld::light_state::LightLevel;
use base::overworld::liquid_state::LiquidState;
use base::overworld::position::{BlockPos, ClusterPos};
use base::overworld::raw_cluster::{BlockDataImpl, LightType, RawCluster};
use base::overworld::ReadOnlyOverworld;
use base::overworld::{block, block_component, raw_cluster, LoadedClusters};
use base::overworld::{Overworld, OverworldOrchestrator};
use base::physics::aabb::{AABBRayIntersection, AABB};
use base::physics::MOTION_EPSILON;
use base::registry::Registry;
use common::glm::{DVec3, I64Vec3, U8Vec4, Vec2, Vec3};
use common::parking_lot::{Mutex, RwLock};
use common::rayon::prelude::*;
use common::resource_file::ResourceFile;
use common::threading::SafeThreadPool;
use common::types::{HashMap, HashSet};
use common::{glm, MO_RELAXED};
use engine::ecs::component;
use engine::ecs::component::render_config::RenderLayer;
use engine::ecs::component::simple_text::{StyledString, TextHAlign, TextStyle};
use engine::ecs::component::ui::{Sizing, UIEventHandlerC, UILayoutC};
use engine::ecs::component::{MeshRenderConfigC, SimpleTextC, TransformC, VertexMeshC};
use engine::module::input::Input;
use engine::module::main_renderer;
use engine::module::main_renderer::{camera, MainRenderer, SimpleObject, VertexMeshObject};
use engine::module::scene::Scene;
use engine::module::text_renderer::{FontSet, RawTextObject, TextRenderer};
use engine::module::ui::{UIObject, UIRenderer};
use engine::module::ui_interaction_manager::UIInteractionManager;
use engine::utils::wsi::find_best_video_mode;
use engine::winit::dpi::LogicalSize;
use engine::winit::event::{MouseButton, VirtualKeyCode};
use engine::winit::event_loop::{ControlFlow, EventLoop};
use engine::winit::window::{CursorGrabMode, Fullscreen, Window, WindowBuilder};
use engine::{winit, Application, Engine, EngineContext};
use entity_data::{AnyState, EntityId};
use std::any::Any;
use std::collections::hash_map;
use std::f32::consts::FRAC_PI_2;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::{atomic, Arc};
use std::thread;
use std::time::{Duration, Instant};

const PROGRAM_NAME: &str = "project-107cf1";
const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);
const PLAYER_CAMERA_OFFSET: DVec3 = DVec3::new(0.0, 0.625, 0.0);

pub struct Game {
    resources: Arc<ResourceFile>,
    registry: Arc<MainRegistry>,
    res_map: Arc<DefaultResourceMapping>,
    default_queue: Arc<SafeThreadPool>,

    cursor_rel: (f64, f64),
    tick_timer: Option<IntervalTimer>,

    main_state: Arc<Mutex<MainState>>,
    overworld_renderer: Option<Arc<Mutex<OverworldRenderer>>>,

    cursor_grab: bool,
    root_entity: EntityId,
}

impl Game {
    const MOUSE_SENSITIVITY: f64 = 0.2;

    pub fn init() -> Game {
        let resources = ResourceFile::open("resources").unwrap();
        let main_registry = MainRegistry::init();
        let res_map = DefaultResourceMapping::init(&main_registry, &resources);

        let overworld = Overworld::new(&main_registry, 0);

        let spawn_point = overworld.interface().generator().gen_spawn_point();
        // dbg!(spawn_point);

        // crate::proto::make_world_prototype_image(overworld.generator());
        // crate::proto::make_climate_graph_image(main_registry.registry());

        let player_pos = glm::convert(spawn_point.0);
        // self.player_pos = DVec3::new(0.5, 64.0, 0.5);

        let mut overworld_orchestrator = OverworldOrchestrator::new(&overworld);
        overworld_orchestrator.set_xz_render_distance(256);
        // overworld_orchestrator.set_xz_render_distance(1024);
        overworld_orchestrator.set_y_render_distance(256);
        overworld_orchestrator.set_stream_pos(player_pos);

        let main_state = Arc::new(Mutex::new(MainState {
            last_tick_start: Instant::now(),
            tick_count: 0,
            overworld,
            overworld_orchestrator: Arc::new(Mutex::new(overworld_orchestrator)),
            res_map: Arc::clone(&res_map),
            player_pos,
            player_orientation: Default::default(),
            player_direction: Default::default(),
            player_aabb: AABB::from_size(DVec3::new(0.6, 1.75, 0.6)),
            fall_time: 0.0,
            curr_jump_force: 0.0,
            look_at_block: None,
            block_set_cooldown: 0.0,
            do_set_block: false,
            curr_block: main_registry.block_test.into_any(),
            set_water: false,
            change_stream_pos: true,
            player_collision_enabled: false,
            do_remove_block: false,
        }));

        // TODO: save tick_count_state to disk

        let program = Game {
            resources,
            registry: Arc::clone(&main_registry),
            res_map,
            default_queue: default_queue().unwrap(),
            cursor_rel: (0.0, 0.0),
            tick_timer: None,
            main_state,
            overworld_renderer: None,
            cursor_grab: true,
            root_entity: Default::default(),
        };
        program
    }

    pub fn grab_cursor(&mut self, window: &Window, enabled: bool, ctx: &EngineContext) {
        self.cursor_grab = enabled;

        if self.cursor_grab {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }

        window.set_cursor_visible(!self.cursor_grab);

        let mut ui_interactor = ctx.module_mut::<UIInteractionManager>();
        ui_interactor.set_active(!self.cursor_grab);
    }
}

// TODO: Make Application this a module

impl Application for Game {
    fn on_engine_start(&mut self, event_loop: &EventLoop<()>) -> Window {
        let window = WindowBuilder::new()
            .with_title(PROGRAM_NAME)
            .with_inner_size(winit::dpi::LogicalSize::new(DEF_WINDOW_SIZE.0, DEF_WINDOW_SIZE.1))
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

        window
    }

    fn initialize_engine(&mut self, ctx: &EngineContext) {
        let scene = Scene::new();
        ctx.register_module(scene);

        self.root_entity = *ctx
            .module_mut::<Scene>()
            .add_object(None, SimpleObject::new())
            .unwrap();

        let mut renderer = MainRenderer::new(
            "VulkanRenderer",
            &*ctx.window(),
            main_renderer::Settings {
                fps_limit: main_renderer::FPSLimit::VSync,
                prefer_triple_buffering: true,
                textures_mipmaps: true,
                texture_quality: main_renderer::TextureQuality::STANDARD,
                textures_max_anisotropy: 1.0,
            },
            512,
            |adapter| 0,
            self.root_entity,
            ctx,
        );
        ctx.register_module(renderer);

        ctx.register_module(TextRenderer::new(ctx));
        ctx.register_module(UIRenderer::new(ctx, self.root_entity));
        ctx.register_module(UIInteractionManager::new(ctx));
        ctx.register_module(Input::new());

        register_ui_elements(ctx);

        // ------------------------------------------------------------------------------------------------

        self.grab_cursor(&*ctx.window(), true, ctx);

        // ------------------------------------------------------------------------------------------------

        let mat_pipelines;
        {
            mat_pipelines = material_pipelines::create(&self.resources, &ctx);

            let mut renderer = ctx.module_mut::<MainRenderer>();

            for (index, (ty, res_ref)) in self.res_map.storage().textures().iter().enumerate() {
                renderer.load_texture_into_atlas(index as u32, *ty, res_ref.clone());
            }

            for (id, material) in self.res_map.storage().materials().iter().enumerate() {
                renderer.set_material(id as u32, material.info());
            }
        }

        let mut renderer = ctx.module_mut::<MainRenderer>();
        let state = self.main_state.lock();

        let overworld_renderer = OverworldRenderer::new(
            Arc::clone(renderer.device()),
            mat_pipelines.cluster,
            Arc::clone(self.registry.registry()),
            Arc::clone(self.res_map.storage()),
            Arc::clone(state.overworld_orchestrator.lock().loaded_clusters()),
            self.root_entity,
        );
        drop(state);

        self.overworld_renderer = Some(Arc::new(Mutex::new(overworld_renderer)));

        {
            let main_state = Arc::clone(&self.main_state);
            let overworld_renderer = Arc::clone(&self.overworld_renderer.as_ref().unwrap());

            self.tick_timer = Some(IntervalTimer::start(
                Duration::from_millis(20),
                VirtualProcessor::new(&self.default_queue),
                move || {
                    let t0 = Instant::now();
                    on_tick(Arc::clone(&main_state), Arc::clone(&overworld_renderer));
                    let t1 = Instant::now();
                    // println!("tick_inner {}", (t1 - t0).as_millis());
                },
            ));
        }
        drop(renderer);

        // -------------------------------------------------------
        let mut ui_ctx = UIContext::new(ctx, &self.resources);

        let mut text_renderer = ctx.module_mut::<TextRenderer>();
        let font_id = text_renderer.register_font(
            FontSet::from_bytes(
                include_bytes!("../res/fonts/Romanesco-Regular.ttf").to_vec(),
                None,
            )
            .unwrap(),
        );
        drop(text_renderer);

        let player_pos = self.main_state.lock().player_pos;
        let text = ui_ctx.scene().add_object(
            Some(self.root_entity),
            RawTextObject::new(
                TransformC::new().with_position(DVec3::new(player_pos.x, player_pos.y + 60.0, player_pos.z)),
                SimpleTextC::new(mat_pipelines.text_3d)
                    .with_text(StyledString::new(
                        "Govno, my is Gmine",
                        TextStyle::new().with_font(font_id).with_font_size(0.5),
                    ))
                    .with_max_width(3.0)
                    .with_h_align(TextHAlign::LEFT),
            ),
        );

        let ui_renderer = ctx.module_mut::<UIRenderer>();
        let root_ui_entity = *ui_renderer.root_ui_entity();
        drop(ui_renderer);

        ui::make_main_menu_screen(&mut ui_ctx, &root_ui_entity);
    }

    fn on_update(&mut self, delta_time: f64, ctx: &EngineContext) {
        let mut renderer = ctx.module_mut::<MainRenderer>();
        let input = ctx.module::<Input>();

        let kb = input.keyboard();
        let mut curr_state = self.main_state.lock();
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
            if !curr_state.player_collision_enabled {
                vel_up_down += 1;
            }
            curr_state.curr_jump_force = 8.0;
        }
        if kb.is_key_pressed(VirtualKeyCode::LShift) {
            vel_up_down -= 1;
        }

        let prev_pos = curr_state.player_pos;
        let ms = MOVEMENT_SPEED * delta_time;
        let mut motion_delta = DVec3::default();

        // Handle translation
        motion_delta.y += vel_up_down as f64 * ms;
        motion_delta += camera::move_xz(
            curr_state.player_orientation,
            vel_front_back as f64 * ms,
            vel_left_right as f64 * ms,
        );

        let mut new_jump_force = curr_state.curr_jump_force;
        let mut new_fall_time = curr_state.fall_time;

        if curr_state.player_collision_enabled {
            let mut blocks = curr_state.overworld.access();
            if blocks
                .get_block(&BlockPos::from_f64(&curr_state.player_pos))
                .is_some()
            {
                // Free fall
                motion_delta.y -= (base::physics::G_ACCEL * curr_state.fall_time) * delta_time;

                // Jump force
                motion_delta.y += new_jump_force * delta_time;
            }
            drop(blocks);

            let new_pos = curr_state
                .overworld
                .move_entity(prev_pos, motion_delta, &curr_state.player_aabb);

            if new_pos.y.abs_diff_eq(&prev_pos.y, MOTION_EPSILON) {
                // Note: the ground is reached
                new_jump_force = 0.0;
                // println!("GROUND HIT");
                new_fall_time = delta_time;
            } else {
                new_fall_time += delta_time;
            }

            curr_state.player_pos = new_pos;
        } else {
            curr_state.player_pos += motion_delta;
        }
        curr_state.fall_time = new_fall_time;
        curr_state.curr_jump_force = new_jump_force;

        {
            let camera = renderer.active_camera_mut();

            // Handle rotation
            let cursor_offset = Vec2::new(self.cursor_rel.0 as f32, self.cursor_rel.1 as f32)
                * (Self::MOUSE_SENSITIVITY * delta_time) as f32;

            let mut rotation = *camera.rotation();
            rotation.x = (rotation.x + cursor_offset.y).clamp(-FRAC_PI_2, FRAC_PI_2);
            rotation.y += cursor_offset.x;
            // rotation.y += delta_time as f32;

            camera.set_rotation(rotation);
            camera.set_position(curr_state.player_pos + PLAYER_CAMERA_OFFSET);
            self.cursor_rel = (0.0, 0.0);

            curr_state.player_orientation = rotation;
            curr_state.player_direction = camera.direction();
        }

        // Handle Look-at
        {
            let cam_pos = curr_state.player_pos + PLAYER_CAMERA_OFFSET;
            let cam_dir = curr_state.player_direction;
            let mut access = curr_state.overworld.access();
            curr_state.look_at_block = access.get_block_at_ray(&cam_pos, &glm::convert(cam_dir), 3.0);
        }

        curr_state.do_set_block = input.mouse().is_button_pressed(MouseButton::Left);
        curr_state.do_remove_block = input.mouse().is_button_pressed(MouseButton::Right);

        drop(curr_state);
        drop(renderer);
        drop(input);

        {
            let mut overworld_renderer = self.overworld_renderer.as_ref().unwrap().lock();
            overworld_renderer.update_scene(&mut ctx.module_mut::<Scene>());
        }
    }

    fn on_event(
        &mut self,
        event: winit::event::Event<()>,
        main_window: &Window,
        control_flow: &mut ControlFlow,
        ctx: &EngineContext,
    ) {
        use engine::winit::event::{DeviceEvent, ElementState, Event, WindowEvent};

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
                        let mut curr_state = self.main_state.lock_arc();

                        match input.virtual_keycode.unwrap() {
                            VirtualKeyCode::Escape => {
                                *control_flow = ControlFlow::Exit;
                            }
                            VirtualKeyCode::L => {
                                curr_state.change_stream_pos = !curr_state.change_stream_pos;
                            }
                            VirtualKeyCode::C => {
                                curr_state.player_collision_enabled = !curr_state.player_collision_enabled;
                            }
                            VirtualKeyCode::F11 => {
                                if let Some(_) = main_window.fullscreen() {
                                    main_window.set_fullscreen(None);
                                } else {
                                    let mode = find_best_video_mode(&main_window.current_monitor().unwrap());
                                    main_window.set_fullscreen(Some(Fullscreen::Exclusive(mode)))
                                }
                            }
                            VirtualKeyCode::P => {
                                println!("{}", curr_state.player_pos);
                                println!(
                                    "{:?}",
                                    curr_state
                                        .player_pos
                                        .map(|v| v.rem_euclid(RawCluster::SIZE as f64))
                                );
                            }
                            VirtualKeyCode::T => {
                                self.grab_cursor(main_window, !self.cursor_grab, &ctx);
                            }
                            VirtualKeyCode::J => {
                                // TODO: move into on_tick
                                // let camera = renderer.active_camera();
                                // let mut access = curr_state.overworld.access();
                                // let block = access.get_block_at_ray(
                                //     &camera.position(),
                                //     &glm::convert(camera.direction()),
                                //     7.0,
                                // );
                                // if let Some((pos, facing)) = block {
                                //     let pos = pos.offset_i32(&Facing::PositiveY.direction());
                                //     let data = access.get_block(&pos);
                                //
                                //     if let Some(data) = data {
                                //         let level = data.liquid_state().level();
                                //         println!("LIQ LEVEL: {level}");
                                //     }
                                // }
                            }
                            VirtualKeyCode::Key1 => {
                                curr_state.curr_block = self.registry.block_test.into_any();
                                curr_state.set_water = false;
                            }
                            VirtualKeyCode::Key2 => {
                                curr_state.curr_block = self.registry.block_glow.into_any();
                                curr_state.set_water = false;
                            }
                            VirtualKeyCode::Key3 => {
                                curr_state.set_water = true;
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

#[derive(Clone)]
struct MainState {
    last_tick_start: Instant,
    tick_count: u64,
    overworld: Arc<Overworld>,
    overworld_orchestrator: Arc<Mutex<OverworldOrchestrator>>,
    res_map: Arc<DefaultResourceMapping>,

    player_pos: DVec3,
    player_orientation: Vec3,
    player_direction: Vec3,
    player_aabb: AABB,
    fall_time: f64,
    curr_jump_force: f64,
    look_at_block: Option<(BlockPos, Facing)>,
    curr_block: AnyBlockState,
    set_water: bool,

    change_stream_pos: bool,
    player_collision_enabled: bool,
    block_set_cooldown: f64,
    do_set_block: bool,
    do_remove_block: bool,
}

const MOVEMENT_SPEED: f64 = 16.0;

fn on_tick(main_state: Arc<Mutex<MainState>>, overworld_renderer: Arc<Mutex<OverworldRenderer>>) {
    let curr_state = main_state.lock().clone();

    let curr_tick = curr_state.tick_count;
    let mut new_actions = OverworldActionsStorage::new();

    player_on_update(&main_state, &mut new_actions);

    let update_res = base::on_tick(
        curr_tick,
        &curr_state.overworld.main_registry().registry(),
        &mut curr_state.overworld_orchestrator.lock(),
        &new_actions,
    );

    let mut overworld_renderer = overworld_renderer.lock();
    overworld_renderer.update(curr_state.player_pos, &update_res);

    main_state.lock().tick_count += 1;
}

fn player_on_update(main_state: &Arc<Mutex<MainState>>, new_actions: &mut OverworldActionsStorage) {
    let start_t = Instant::now();
    let curr_state = main_state.lock().clone();
    let registry = curr_state.overworld.main_registry();
    let delta_time = (start_t - curr_state.last_tick_start).as_secs_f64();

    let mut new_block_set_cooldown = curr_state.block_set_cooldown;

    // Set block on mouse button click
    if curr_state.block_set_cooldown == 0.0 {
        if curr_state.do_set_block {
            if let Some((pos, facing)) = curr_state.look_at_block {
                let dir: I64Vec3 = glm::convert(*facing.direction());
                let set_pos = pos.offset(&dir);

                if curr_state.set_water {
                    new_actions.set_liquid(set_pos, LiquidState::source(curr_state.res_map.material_water()));
                } else {
                    new_actions.set_block(set_pos, curr_state.curr_block.clone());

                    if curr_state.curr_block.block_id == registry.block_glow.block_id {
                        new_actions.set_light_source(
                            set_pos,
                            LightLevel::from_intensity(10),
                            LightType::Regular,
                        );
                    } else {
                    }
                }

                new_block_set_cooldown = 0.15;
            }
        } else if curr_state.do_remove_block {
            if let Some(inter) = curr_state.look_at_block {
                new_actions.set_block(inter.0, registry.block_empty);
                new_actions.set_light_source(inter.0, LightLevel::ZERO, LightType::Regular);

                new_block_set_cooldown = 0.15;
            }
        }
    } else {
        new_block_set_cooldown = (curr_state.block_set_cooldown - delta_time).max(0.0);
    }

    if curr_state.change_stream_pos {
        curr_state
            .overworld_orchestrator
            .lock()
            .set_stream_pos(curr_state.player_pos);
    }

    let mut new_state = main_state.lock();
    new_state.last_tick_start = start_t;
    new_state.block_set_cooldown = new_block_set_cooldown;
}
