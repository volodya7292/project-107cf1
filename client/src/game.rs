mod ui;

use crate::default_resource_mapping::DefaultResourceMapping;
use crate::rendering::material_pipelines;
use crate::rendering::material_pipelines::MaterialPipelines;
use crate::rendering::overworld_renderer::OverworldRenderer;
use crate::rendering::ui::image::ImageImpl;
use crate::rendering::ui::register_ui_elements;
use approx::AbsDiffEq;
use base::check_block;
use base::execution::default_queue;
use base::execution::timer::IntervalTimer;
use base::execution::virtual_processor::VirtualProcessor;
use base::main_registry::MainRegistry;
use base::overworld::accessor::ReadOnlyOverworldAccessorImpl;
use base::overworld::actions_storage::OverworldActionsStorage;
use base::overworld::block::AnyBlockState;
use base::overworld::facing::Facing;
use base::overworld::interface::local_interface::LocalOverworldInterface;
use base::overworld::light_state::LightLevel;
use base::overworld::liquid_state::LiquidState;
use base::overworld::position::BlockPos;
use base::overworld::raw_cluster::{BlockDataImpl, LightType};
use base::overworld::{Overworld, OverworldOrchestrator, OverworldState, PlayerState};
use base::physics::aabb::AABB;
use base::physics::{calc_acceleration, calc_force, G_ACCEL, MOTION_EPSILON};
use common::glm::{DVec3, I64Vec3, Vec2, Vec3};
use common::lrc::{Lrc, LrcExtSized, OwnedRefMut};
use common::parking_lot::Mutex;
use common::resource_file::{BufferedResourceReader, ResourceFile};
use common::threading::SafeThreadPool;
use common::{glm, image};
use engine::event::{WSIEvent, WSIKeyboardInput};
use engine::module::input::{Input, Keyboard};
use engine::module::main_renderer::{camera, MainRenderer, PostProcess, WrapperObject};
use engine::module::scene::Scene;
use engine::module::text_renderer::TextRenderer;
use engine::module::ui::reactive::UIReactor;
use engine::module::ui::UIRenderer;
use engine::module::ui_interaction_manager::UIInteractionManager;
use engine::module::{main_renderer, EngineModule};
use engine::utils::transition::{AnimatedValue, TransitionTarget};
use engine::utils::wsi::find_best_video_mode;
use engine::vkw::utils::GLSLBool;
use engine::winit::event::{MouseButton, VirtualKeyCode};
use engine::winit::event_loop::EventLoop;
use engine::winit::window::{CursorGrabMode, Fullscreen, Window, WindowBuilder};
use engine::{winit, EngineContext};
use entity_data::EntityId;
use std::cell::RefMut;
use std::f32::consts::FRAC_PI_2;
use std::f64::consts::PI;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use self::ui::common::ui_root_states;

const PROGRAM_NAME: &str = "project-107cf1";
const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);
const PLAYER_CAMERA_OFFSET: DVec3 = DVec3::new(0.0, 0.625, 0.0);

pub trait SceneGameExt {
    fn resources(&self) -> Arc<BufferedResourceReader>;
}

impl SceneGameExt for Scene {
    fn resources(&self) -> Arc<BufferedResourceReader> {
        Arc::clone(&self.resource::<Arc<BufferedResourceReader>>())
    }
}

pub trait EngineCtxGameExt {
    fn app(&self) -> OwnedRefMut<dyn EngineModule, MainApp>;
    fn scene(&self) -> OwnedRefMut<dyn EngineModule, Scene>;
    fn resource_image(
        scene: &Scene,
        filename: &str,
    ) -> Result<Arc<image::RgbaImage>, common::resource_file::Error>;
}

impl EngineCtxGameExt for EngineContext<'_> {
    fn app(&self) -> OwnedRefMut<dyn EngineModule, MainApp> {
        self.module_mut::<MainApp>()
    }

    fn scene(&self) -> OwnedRefMut<dyn EngineModule, Scene> {
        self.module_mut::<Scene>()
    }

    fn resource_image(
        scene: &Scene,
        filename: &str,
    ) -> Result<Arc<image::RgbaImage>, common::resource_file::Error> {
        let resources = scene.resources();
        resources.get_image(filename)
    }
}

struct GameSettings {
    fov_y: f32,
}

pub struct MainApp {
    resources: Arc<BufferedResourceReader>,
    main_registry: Arc<MainRegistry>,
    res_map: Arc<DefaultResourceMapping>,
    default_queue: Arc<SafeThreadPool>,

    cursor_rel: Vec2,
    tick_timer: Option<IntervalTimer>,

    material_pipelines: MaterialPipelines,
    game_state: Option<Arc<Mutex<GameProcessState>>>,
    settings: GameSettings,
    post_liquid_uniforms: Arc<Mutex<PostProcessLiquidUniforms>>,

    cursor_grab: bool,
    root_entity: EntityId,
    ui_reactor: Lrc<UIReactor>,
}

const WALK_VELOCITY: f64 = 3.0;

fn calc_bobbing_displacement(walk_time: f64, walk_vel: f64, camera_orientation: Vec3) -> DVec3 {
    const BOBBING_HEIGHT: f64 = 0.05;
    const BOBBING_WIDTH: f64 = 0.04;

    let bobbing_freq: f64 = 0.7 * walk_vel;
    let wave = (walk_time * 2.0 * PI * bobbing_freq * 0.5).sin();

    let disp_y_norm = wave.powi(2);

    let mut displacement = DVec3::zeros();
    displacement += camera::move_xz(camera_orientation, 0.0, (wave) * BOBBING_WIDTH * 0.5);
    displacement.y -= disp_y_norm * BOBBING_HEIGHT;

    displacement
}

#[derive(Default, Copy, Clone)]
#[repr(C)]
struct PostProcessLiquidUniforms {
    enabled: GLSLBool,
}

impl MainApp {
    const MOUSE_SENSITIVITY: f64 = 0.2;
    const PLAYER_MASS: f32 = 80.0; // health won't automatically regenerate past this value
    const DAY_CYCLE_TIME_INTERVAL: f64 = 60.0 * 30.0; // 30 minutes
    const SATIETY_IDLE_DECREASE_SPEED: f64 = 1.0 / Self::DAY_CYCLE_TIME_INTERVAL;
    const HEALTH_MAX_REGEN_SPEED: f64 = 1.0 / 200.0;
    const HEALTH_REGEN_THRESHOLD: f64 = 0.5; // health won't automatically regenerate past this value
    const MAX_SAFE_PLAYER_VELOCITY: f32 = 12.0;
    const FATAL_PLAYER_VELOCITY: f32 = 21.0;

    pub fn init(ctx: &EngineContext) {
        ctx.register_module(Scene::new());

        let root_entity = *ctx
            .module_mut::<Scene>()
            .add_object(None, WrapperObject::new())
            .unwrap();

        let post_liquid_uniforms = Arc::new(Mutex::new(PostProcessLiquidUniforms::default()));
        let renderer_post_processes = vec![
            PostProcess {
                name: "sky".to_string(),
                shader_code: include_bytes!("../res/shaders/post_sky.frag.spv").to_vec(),
                uniform_data: Arc::new(Mutex::new(())),
            },
            PostProcess {
                name: "liquid_distortions".to_string(),
                shader_code: include_bytes!("../res/shaders/post_liquid_distortions.frag.spv").to_vec(),
                uniform_data: post_liquid_uniforms.clone(),
            },
        ];
        let renderer = MainRenderer::new(
            "VulkanRenderer",
            &ctx.window(),
            main_renderer::Settings {
                fps_limit: main_renderer::FPSLimit::VSync,
                prefer_triple_buffering: true,
                textures_mipmaps: true,
                texture_quality: main_renderer::TextureQuality::STANDARD,
                textures_max_anisotropy: 1.0,
            },
            512,
            |adapter| 0,
            root_entity,
            ctx,
            renderer_post_processes,
        );
        ctx.register_module(renderer);

        ctx.register_module(TextRenderer::new(ctx));
        ctx.register_module(UIRenderer::new(ctx, root_entity));
        ctx.register_module(UIInteractionManager::new(ctx));
        ctx.register_module(Input::new());

        register_ui_elements(ctx);

        let resources = BufferedResourceReader::new(ResourceFile::open("resources").unwrap());

        let main_registry = MainRegistry::init();
        let res_map = DefaultResourceMapping::init(&main_registry, &resources);

        let mat_pipelines;
        {
            mat_pipelines = material_pipelines::create(resources.file(), ctx);

            let mut renderer = ctx.module_mut::<MainRenderer>();

            for (index, (ty, res_ref)) in res_map.storage().textures().iter().enumerate() {
                renderer.load_texture_into_atlas(index as u32, *ty, res_ref.clone());
            }

            for (id, material) in res_map.storage().materials().iter().enumerate() {
                renderer.set_material(id as u32, material.info());
            }
        }

        let ui_root_element = {
            let ui_renderer = ctx.module_mut::<UIRenderer>();
            *ui_renderer.root_ui_entity()
        };

        let ui_reactor = { UIReactor::new(move |ctx| ui::overlay_root(ctx, ui_root_element)) };

        let res =
            engine::gltf::load_simple_gltf(&resources.get("models/material_item_blob.glb").unwrap()).unwrap();
        println!(
            "{} {:?} {:?}",
            res.positions.len(),
            res.normals.map(|v| v.len()),
            res.indices.map(|v| v.len()),
        );

        let game = MainApp {
            resources: Arc::clone(&resources),
            main_registry: Arc::clone(&main_registry),
            res_map,
            default_queue: default_queue().unwrap(),
            cursor_rel: Default::default(),
            tick_timer: None,
            material_pipelines: mat_pipelines,
            game_state: None,
            settings: GameSettings { fov_y: FRAC_PI_2 },
            post_liquid_uniforms,
            cursor_grab: false,
            root_entity,
            ui_reactor: Lrc::wrap(ui_reactor),
        };
        ctx.register_module(game);

        ctx.module::<Scene>().register_resource(resources);

        if !Self::data_dir().exists() {
            fs::create_dir(Self::data_dir()).unwrap();
        }
        if !Self::worlds_dir().exists() {
            fs::create_dir(Self::worlds_dir()).unwrap();
        }
    }

    pub fn create_window(event_loop: &EventLoop<()>) -> Window {
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

    pub fn resources(&self) -> &Arc<BufferedResourceReader> {
        &self.resources
    }

    pub fn ui_reactor(&self) -> RefMut<UIReactor> {
        self.ui_reactor.borrow_mut()
    }

    pub fn grab_cursor(&mut self, window: &Window, enabled: bool, ctx: &EngineContext) {
        if self.cursor_grab == enabled {
            return;
        }
        self.cursor_grab = enabled;

        if enabled {
            window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
                .unwrap();
        } else {
            window.set_cursor_grab(CursorGrabMode::None).unwrap();
        }

        window.set_cursor_visible(!enabled);
    }

    pub fn is_in_game(&self) -> bool {
        self.game_state.is_some()
    }

    pub fn is_cursor_grabbed(&self) -> bool {
        self.cursor_grab
    }

    pub fn show_main_menu(&self, visible: bool) {
        let ui_reactor = self.ui_reactor.borrow();
        if let Some(state) = ui_reactor.root_state(&ui_root_states::MENU_VISIBLE) {
            state.update(visible);
        }
    }

    pub fn is_menu_visible(&self) -> Option<bool> {
        let ui_reactor = self.ui_reactor.borrow();
        ui_reactor
            .root_state(&ui_root_states::MENU_VISIBLE)
            .map(|v| *v.value())
    }

    pub fn make_world_path(overworld_name: &str) -> PathBuf {
        Self::worlds_dir().join(overworld_name)
    }

    pub fn create_overworld(&mut self, overworld_name: &str, seed_str: &str) {
        let world_path = Self::make_world_path(overworld_name);

        let seed_digest = common::ring::digest::digest(&common::ring::digest::SHA256, seed_str.as_bytes());
        let seed = u64::from_le_bytes((seed_digest.as_ref()[..8]).try_into().unwrap());

        LocalOverworldInterface::create_overworld(
            world_path,
            OverworldState {
                seed,
                tick_count: 0,
                player_state: PlayerState::new(),
            },
        );
    }

    pub fn delete_overworld(&mut self, overworld_name: &str) {
        let world_path = Self::make_world_path(overworld_name);
        let _ = fs::remove_dir_all(world_path);
    }

    pub fn enter_overworld(&mut self, ctx: &EngineContext, overworld_name: &str) {
        let world_path = Self::make_world_path(overworld_name);

        let interface = LocalOverworldInterface::new(world_path, &self.main_registry);
        let overworld = Overworld::new(&self.main_registry, Arc::new(interface));

        // crate::proto::make_world_prototype_image(overworld.generator());
        // crate::proto::make_climate_graph_image(main_registry.registry());

        let params = overworld.interface().persisted_state();

        let player_pos = params.player_state().position().unwrap_or_else(|| {
            let block_spawn_point = overworld.interface().generator().gen_spawn_point().0;
            let real_spawn_point: DVec3 = glm::convert(block_spawn_point);
            real_spawn_point + glm::vec3(0.5, 0.5, 0.5)
        });
        let player_velocity = params.player_state().velocity();

        let mut overworld_orchestrator = OverworldOrchestrator::new(&overworld);
        overworld_orchestrator.set_xz_render_distance(256);
        // overworld_orchestrator.set_xz_render_distance(1024);
        overworld_orchestrator.set_y_render_distance(256);
        overworld_orchestrator.set_stream_pos(player_pos);
        let overworld_orchestrator = Arc::new(Mutex::new(overworld_orchestrator));

        let overworld_renderer = OverworldRenderer::new(
            Arc::clone(ctx.module::<MainRenderer>().device()),
            self.material_pipelines.cluster,
            Arc::clone(self.main_registry.registry()),
            Arc::clone(self.res_map.storage()),
            Arc::clone(overworld_orchestrator.lock().loaded_clusters()),
            self.root_entity,
        );
        let overworld_renderer = Arc::new(Mutex::new(overworld_renderer));

        // TODO: save tick_count_state to disk

        let new_game_state = Arc::new(Mutex::new(GameProcessState {
            last_tick_start: Instant::now(),
            tick_count: params.tick_count,
            overworld,
            overworld_orchestrator: Some(overworld_orchestrator),
            overworld_renderer: Some(Arc::clone(&overworld_renderer)),
            res_map: Arc::clone(&self.res_map),
            player_aabb: AABB::from_size(DVec3::new(0.6, 1.75, 0.6)),
            fall_time: 0.0,
            walk_time: 0.0,
            walk_velocity: Default::default(),
            bobbing_offset: Default::default(),
            curr_jump_force: 0.0,
            look_at_block: None,
            block_set_cooldown: 0.0,
            do_set_block: false,
            curr_block: self.main_registry.block_test.into_any(),
            set_water: false,
            change_stream_pos: true,
            player_collision_enabled: true,
            do_remove_block: false,
        }));

        self.game_state = Some(Arc::clone(&new_game_state));
        self.tick_timer = Some(IntervalTimer::start(
            Duration::from_millis(20),
            VirtualProcessor::new(&self.default_queue),
            move || {
                let t0 = Instant::now();
                on_tick(new_game_state.clone(), overworld_renderer.clone());
                let t1 = Instant::now();
                // println!("tick_inner {}", (t1 - t0).as_millis());
            },
        ));

        self.show_main_menu(false);
    }

    pub fn exit_overworld(&mut self, ctx: &EngineContext) {
        self.tick_timer.take().unwrap().stop_and_join();
        println!("World timer stopped.");

        let game_state = self.game_state.take().unwrap();
        {
            let mut game_state = game_state.lock();

            let overworld_orchestrator = game_state.overworld_orchestrator.take().unwrap();
            overworld_orchestrator.lock().stop_and_join();
            drop(overworld_orchestrator);

            let overworld_renderer = game_state.overworld_renderer.take().unwrap();
            overworld_renderer.lock().cleanup_everything(&mut ctx.scene());
            drop(overworld_renderer);

            println!("Overworld engine stopped. Saving changes...");

            let local_interface = game_state
                .overworld
                .interface()
                .as_any()
                .downcast_ref::<LocalOverworldInterface>()
                .unwrap();
            local_interface.stop_and_commit_changes();
        }

        println!("Cleaning up...");
        drop(game_state);
    }

    pub fn respawn_player(&self) {
        let game_state = self.game_state.as_ref().unwrap().lock();

        let player_pos = {
            let block_spawn_point = game_state.overworld.interface().generator().gen_spawn_point().0;
            let real_spawn_point: DVec3 = glm::convert(block_spawn_point);
            real_spawn_point + glm::vec3(0.5, 0.5, 0.5)
        };

        game_state.overworld.interface().update_persisted_state(&|state| {
            state.update_player_state(|p_state| {
                p_state.set_health(1.0);
                p_state.set_satiety(1.0);
                p_state.set_position(player_pos);
                p_state.set_velocity(Vec3::zeros());
            });
        });
    }

    pub fn data_dir() -> PathBuf {
        dirs::data_dir().unwrap().join(PROGRAM_NAME)
    }

    pub fn worlds_dir() -> PathBuf {
        Self::data_dir().join("overworlds")
    }

    pub fn get_world_name_list(&self) -> Vec<String> {
        let dir = Self::worlds_dir();
        let contents = dir.read_dir().unwrap();
        contents
            .filter_map(Result::ok)
            .filter(|entry| entry.path().is_dir())
            .filter_map(|entry| entry.file_name().to_str().map(|v| v.to_string()))
            .collect()
    }

    fn on_game_wsi_event(&mut self, main_window: &Window, event: &WSIEvent, ctx: &EngineContext) {
        use engine::winit::event::ElementState;

        let main_state = self.game_state.as_ref().unwrap();
        let player_state = main_state
            .lock()
            .overworld
            .interface()
            .persisted_state()
            .player_state();

        match event {
            WSIEvent::KeyboardInput { input } => {
                let WSIKeyboardInput::Virtual(virtual_keycode, state) = input else {
                    return;
                };

                if *state != ElementState::Released {
                    return;
                }

                let mut curr_state = main_state.lock_arc();

                match virtual_keycode {
                    VirtualKeyCode::L => {
                        curr_state.change_stream_pos = !curr_state.change_stream_pos;
                    }
                    VirtualKeyCode::C => {
                        curr_state.player_collision_enabled = !curr_state.player_collision_enabled;
                    }
                    VirtualKeyCode::P => {
                        println!("{:?}", player_state.position());
                    }
                    VirtualKeyCode::T => {
                        self.grab_cursor(main_window, !self.cursor_grab, ctx);
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
                        curr_state.curr_block = self.main_registry.block_test.into_any();
                        curr_state.set_water = false;
                    }
                    VirtualKeyCode::Key2 => {
                        curr_state.curr_block = self.main_registry.block_glow.into_any();
                        curr_state.set_water = false;
                    }
                    VirtualKeyCode::Key3 => {
                        curr_state.set_water = true;
                    }
                    _ => {}
                }
            }
            WSIEvent::MouseMotion { delta } => {
                self.cursor_rel += glm::convert::<_, Vec2>(*delta);
            }
            _ => {}
        }
    }

    fn calc_player_velocity(kb: &Keyboard, state: &GameProcessState, orientation: Vec3) -> Vec3 {
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
            if !state.player_collision_enabled {
                vel_up_down += 1;
            }
        }
        if kb.is_key_pressed(VirtualKeyCode::LShift) {
            vel_up_down -= 1;
        }

        let horizontal_move: Vec3 = glm::convert(camera::move_xz(
            orientation,
            vel_front_back as f64,
            vel_left_right as f64,
        ));
        let vertical_move = Vec3::new(0.0, vel_up_down as f32, 0.0);

        (horizontal_move + vertical_move) * (WALK_VELOCITY as f32)
    }

    fn on_update_game_input(&mut self, delta_time: f64, ctx: &EngineContext) {
        let Some(main_state) = self.game_state.as_ref() else {
            return;
        };

        let input = ctx.module::<Input>();
        let kb = input.keyboard();

        let mut curr_state = main_state.lock();
        let player_state = curr_state.overworld.interface().persisted_state().player_state();

        if player_state.is_dead() {
            return;
        }
        let Some(player_position) = player_state.position() else {
            return;
        };

        if kb.is_key_pressed(VirtualKeyCode::Space) && curr_state.fall_time == 0.0 {
            curr_state.curr_jump_force = MainApp::PLAYER_MASS * 5.5 / delta_time as f32;
        }

        {
            let move_vel =
                Self::calc_player_velocity(kb, &curr_state, glm::convert(player_state.orientation()));
            curr_state
                .walk_velocity
                .retarget(TransitionTarget::new(move_vel, 0.07));
            curr_state.walk_velocity.advance(delta_time);
        }
        let walk_vel = *curr_state.walk_velocity.current();

        if walk_vel.xz().magnitude() as f64 >= 0.5 {
            // velocity is greater than 0.5 m/s => walking
            let mut renderer = ctx.module_mut::<MainRenderer>();
            let camera = renderer.active_camera_mut();

            let bobbing_offset =
                calc_bobbing_displacement(curr_state.walk_time, WALK_VELOCITY, *camera.rotation());

            const SMOOTHING_TIME: f64 = 0.2;
            let bobbing_offset_smoothed = glm::convert::<_, Vec3>(bobbing_offset)
                * (curr_state.walk_time.min(SMOOTHING_TIME) / SMOOTHING_TIME) as f32;

            curr_state.bobbing_offset.retarget(bobbing_offset_smoothed.into());
            curr_state.walk_time += delta_time;
        } else if curr_state.walk_time != 0.0 {
            curr_state.walk_time = 0.0;
            curr_state
                .bobbing_offset
                .retarget(TransitionTarget::new(Vec3::zeros(), 0.2));
        }
        curr_state.bobbing_offset.advance(delta_time);

        let (mut new_player_velocity, motion_delta) = {
            let g_force = calc_force(MainApp::PLAYER_MASS, Vec3::new(0.0, -G_ACCEL as f32, 0.0));

            let mut total_force = Vec3::zeros();

            if curr_state.player_collision_enabled {
                let mut blocks = curr_state.overworld.access();
                if blocks.get_block(&BlockPos::from_f64(&player_position)).is_some() {
                    let jump_force = Vec3::new(0.0, curr_state.curr_jump_force, 0.0);
                    total_force += g_force + jump_force;
                }
            }

            let new_player_velocity = player_state.velocity()
                + calc_acceleration(total_force, MainApp::PLAYER_MASS) * delta_time as f32;
            let motion_delta = glm::convert::<_, DVec3>(new_player_velocity + walk_vel) * delta_time;

            (new_player_velocity, motion_delta)
        };

        curr_state.curr_jump_force = 0.0;
        let mut new_jump_force = curr_state.curr_jump_force;
        let mut new_player_pos = player_position;
        let mut health_decrease = 0.0;

        if curr_state.player_collision_enabled {
            let prev_pos = player_position;

            let new_pos =
                curr_state
                    .overworld
                    .try_resolve_collisions(prev_pos, motion_delta, &curr_state.player_aabb);

            if new_pos.y.abs_diff_eq(&prev_pos.y, MOTION_EPSILON) {
                if new_player_velocity.y < -Self::MAX_SAFE_PLAYER_VELOCITY {
                    health_decrease = ((new_player_velocity.y.abs() - Self::MAX_SAFE_PLAYER_VELOCITY)
                        / (Self::FATAL_PLAYER_VELOCITY - Self::MAX_SAFE_PLAYER_VELOCITY))
                        as f64;
                }

                // Note: the ground is reached
                new_jump_force = 0.0;
                new_player_velocity = Vec3::zeros();
                curr_state.fall_time = 0.0;
            } else {
                curr_state.fall_time += delta_time;
            }

            new_player_pos = new_pos;
        } else {
            new_player_velocity = Vec3::zeros();
            new_player_pos += motion_delta;
        }
        curr_state.curr_jump_force = new_jump_force;

        let new_player_orientation;
        {
            let mut renderer = ctx.module_mut::<MainRenderer>();
            let camera = renderer.active_camera_mut();

            // Handle rotation
            let cursor_offset = self.cursor_rel * (Self::MOUSE_SENSITIVITY * delta_time) as f32;

            let mut rotation = *camera.rotation();
            rotation.x = (rotation.x + cursor_offset.y).clamp(-FRAC_PI_2, FRAC_PI_2);
            rotation.y += cursor_offset.x;

            let bobbing_offset: DVec3 = glm::convert(*curr_state.bobbing_offset.current());

            camera.set_rotation(rotation);
            camera.set_position(new_player_pos + PLAYER_CAMERA_OFFSET + bobbing_offset);
            self.cursor_rel = Vec2::new(0.0, 0.0);

            new_player_orientation = glm::convert(rotation);
        }

        // Handle Look-at
        {
            let mut renderer = ctx.module_mut::<MainRenderer>();
            let camera = renderer.active_camera_mut();

            let cam_pos = camera.position();
            let cam_dir = camera.direction();

            let mut access = curr_state.overworld.access();
            curr_state.look_at_block = access.get_block_at_ray(cam_pos, &glm::convert(cam_dir), 3.0);
        }

        // Handle inside-block vision filters
        {
            let registry = self.main_registry.registry();
            let reactor = self.ui_reactor();
            let vision_obstructed_state = reactor.root_state(&ui_root_states::VISION_OBSTRUCTED).unwrap();
            let mut access = curr_state.overworld.access();

            let block_pos = BlockPos::from_f64(&(new_player_pos + PLAYER_CAMERA_OFFSET));

            if let Some(data) = access.get_block(&block_pos) {
                let block = registry.get_block(data.block_id()).unwrap();

                vision_obstructed_state
                    .update(!block.transparent() && registry.get_block_model(block.model_id()).is_some());

                let inside_liquid = !data.liquid_state().is_empty();

                {
                    let mut post_liquid = self.post_liquid_uniforms.lock();
                    post_liquid.enabled = inside_liquid as GLSLBool;
                };

                let mut renderer = ctx.module_mut::<MainRenderer>();
                let camera = renderer.active_camera_mut();

                if inside_liquid {
                    camera.set_fovy(self.settings.fov_y * 0.75);
                } else {
                    camera.set_fovy(self.settings.fov_y);
                }
            }
        }

        curr_state.do_set_block = input.mouse().is_button_pressed(MouseButton::Left);
        curr_state.do_remove_block = input.mouse().is_button_pressed(MouseButton::Right);

        curr_state.overworld.interface().update_persisted_state(&|state| {
            state.update_player_state(|p_state| {
                p_state.set_position(new_player_pos);
                p_state.set_orientation(new_player_orientation);
                p_state.set_velocity(new_player_velocity);

                p_state.set_health((p_state.health() - health_decrease).max(0.0));
                // TODO: death screen
            });
        });
    }
}

impl EngineModule for MainApp {
    fn on_start(&mut self, ctx: &EngineContext) {
        // let mut ui_ctx = UIContext::new(ctx, &self.resources);

        // let mut text_renderer = ctx.module_mut::<TextRenderer>();
        // let font_id = text_renderer.register_font(
        //     FontSet::from_bytes(
        //         include_bytes!("../res/fonts/Romanesco-Regular.ttf").to_vec(),
        //         None,
        //     )
        //     .unwrap(),
        // );
        // drop(text_renderer);

        // let player_pos = self.main_state.lock().player_pos;
        // let text = ui_ctx.scene().add_object(
        //     Some(self.root_entity),
        //     RawTextObject::new(
        //         TransformC::new().with_position(DVec3::new(player_pos.x, player_pos.y + 60.0, player_pos.z)),
        //         SimpleTextC::new(self.material_pipelines.text_3d)
        //             .with_text(StyledString::new(
        //                 "Govno, my is Gmine",
        //                 TextStyle::new().with_font(font_id).with_font_size(0.5),
        //             ))
        //             .with_max_width(3.0)
        //             .with_h_align(TextHAlign::LEFT),
        //     ),
        // );
    }

    fn on_update(&mut self, delta_time: f64, ctx: &EngineContext) {
        {
            let ui_reactor = self.ui_reactor.clone();
            ctx.dispatch_callback(move |ctx, _| {
                ui_reactor.borrow_mut().on_update(*ctx, delta_time);
            });
        }

        if let Some(curr_state) = self.game_state.as_ref().map(|v| v.lock()) {
            let reactor = self.ui_reactor();
            // Update HUD
            {
                let player_health_state = reactor.root_state(&ui_root_states::PLAYER_HEALTH).unwrap();
                let player_satiety_state = reactor.root_state(&ui_root_states::PLAYER_SATIETY).unwrap();

                let player_state = curr_state.overworld.interface().persisted_state().player_state();
                player_health_state.update(player_state.health());
                player_satiety_state.update(player_state.satiety());
            }

            // Update debug info
            {
                let fps = 1.0 / delta_time;
                let app_time = ctx.module_last_update_time::<MainApp>();
                let input_time = ctx.module_last_update_time::<Input>();
                let ui_inter_manager_time = ctx.module_last_update_time::<UIInteractionManager>();
                let ui_renderer_time = ctx.module_last_update_time::<UIRenderer>();
                let main_renderer_time = ctx.module_last_update_time::<MainRenderer>();
                let scene_time = ctx.module_last_update_time::<Scene>();

                let renderer_timings = ctx.module::<MainRenderer>().last_timings();

                let debug_info_state = reactor.root_state(&ui_root_states::DEBUG_INFO).unwrap();
                debug_info_state.update(vec![
                    format!("{fps:.1} fps"),
                    format!("MainApp: {:.1} ms", app_time * 1000.0),
                    format!("Input: {:.1} ms", input_time * 1000.0),
                    format!("UIInteractionManager: {:.1} ms", ui_inter_manager_time * 1000.0),
                    format!("UIRenderer: {:.1} ms", ui_renderer_time * 1000.0),
                    format!(
                        "MainRenderer: {:.1} ms: {}",
                        main_renderer_time * 1000.0,
                        renderer_timings
                    ),
                    format!("Scene: {:.1} ms", scene_time * 1000.0),
                ]);
            }
        };

        if self.is_menu_visible().unwrap_or(true) {
            return;
        }

        self.on_update_game_input(delta_time, ctx);

        let Some(main_state) = self.game_state.as_ref() else {
            return;
        };

        let overworld_renderer = Arc::clone(main_state.lock().overworld_renderer.as_ref().unwrap());
        overworld_renderer
            .lock()
            .update_scene(&mut ctx.module_mut::<Scene>());
    }

    fn on_wsi_event(&mut self, main_window: &Window, event: &WSIEvent, ctx: &EngineContext) {
        use engine::winit::event::ElementState;

        if self.is_cursor_grabbed() && self.game_state.is_some() {
            self.on_game_wsi_event(main_window, event, ctx);
        }

        match event {
            WSIEvent::KeyboardInput { input } => {
                let WSIKeyboardInput::Virtual(virtual_keycode, state) = input else {
                    return;
                };
                if *state != ElementState::Released {
                    return;
                }

                match virtual_keycode {
                    VirtualKeyCode::Escape => {
                        if self.is_in_game() {
                            self.show_main_menu(!self.is_menu_visible().unwrap_or(false));
                        }
                    }
                    VirtualKeyCode::F11 => {
                        if let Some(_) = main_window.fullscreen() {
                            main_window.set_fullscreen(None);
                        } else {
                            let mode = find_best_video_mode(&main_window.current_monitor().unwrap());
                            main_window.set_fullscreen(Some(Fullscreen::Exclusive(mode)))
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
}

#[derive(Clone)]
struct GameProcessState {
    last_tick_start: Instant,
    tick_count: u64,
    overworld: Arc<Overworld>,
    overworld_orchestrator: Option<Arc<Mutex<OverworldOrchestrator>>>,
    overworld_renderer: Option<Arc<Mutex<OverworldRenderer>>>,
    res_map: Arc<DefaultResourceMapping>,

    player_aabb: AABB,
    fall_time: f64,
    walk_time: f64,
    walk_velocity: AnimatedValue<Vec3>,
    bobbing_offset: AnimatedValue<Vec3>,
    curr_jump_force: f32,
    look_at_block: Option<(BlockPos, Facing)>,
    curr_block: AnyBlockState,
    set_water: bool,

    change_stream_pos: bool,
    player_collision_enabled: bool,
    block_set_cooldown: f64,
    do_set_block: bool,
    do_remove_block: bool,
}

fn on_tick(main_state: Arc<Mutex<GameProcessState>>, overworld_renderer: Arc<Mutex<OverworldRenderer>>) {
    let curr_state = main_state.lock().clone();
    let player_state = curr_state.overworld.interface().persisted_state().player_state();

    let curr_tick = curr_state.tick_count;
    let mut new_actions = OverworldActionsStorage::new();

    player_on_update(&main_state, &mut new_actions);

    let update_res = base::on_tick(
        curr_tick,
        curr_state.overworld.main_registry().registry(),
        &mut curr_state.overworld_orchestrator.as_ref().unwrap().lock(),
        &new_actions,
    );

    let mut overworld_renderer = overworld_renderer.lock();
    if let Some(player_pos) = player_state.position() {
        overworld_renderer.update(player_pos, &update_res);
    }

    main_state.lock().tick_count += 1;

    // Update persisted params
    {
        let tick_count = main_state.lock().tick_count;
        curr_state.overworld.interface().update_persisted_state(&|state| {
            state.tick_count = tick_count;
        });
    }
}

fn player_on_update(main_state: &Arc<Mutex<GameProcessState>>, new_actions: &mut OverworldActionsStorage) {
    let start_t = Instant::now();
    let curr_state = main_state.lock().clone();
    let player_state = curr_state.overworld.interface().persisted_state().player_state();
    let Some(player_pos) = player_state.position() else {
        return;
    };
    let registry = curr_state.overworld.main_registry();
    let delta_time = (start_t - curr_state.last_tick_start).as_secs_f64();

    let mut new_block_set_cooldown = curr_state.block_set_cooldown;
    let overworld = Arc::clone(&main_state.lock().overworld);

    // Set block on mouse button click
    if curr_state.block_set_cooldown == 0.0 {
        if curr_state.do_set_block {
            if let Some((pos, facing)) = curr_state.look_at_block {
                let dir: I64Vec3 = glm::convert(*facing.direction());
                let set_pos = pos.offset(&dir);

                let new_block_global_aabb = AABB::block().translate(&glm::convert(set_pos.0));
                let player_global_aabb = curr_state.player_aabb.translate(&player_pos);

                if curr_state.set_water {
                    overworld
                        .access()
                        .set_liquid_state(&set_pos, LiquidState::source(registry.liquid_water));
                } else if !player_global_aabb.collides_with(&new_block_global_aabb) {
                    overworld.access().update_block(&set_pos, |data| {
                        data.set(curr_state.curr_block.clone());

                        if curr_state.curr_block.block_id == registry.block_glow.block_id {
                            *data.light_source_type_mut() = LightType::Regular;
                            *data.raw_light_source_mut() = LightLevel::from_intensity(10);
                        } else {
                            *data.raw_light_source_mut() = LightLevel::ZERO;
                        }
                    });
                }

                check_block(registry.registry(), &mut overworld.access(), &set_pos);

                new_block_set_cooldown = 0.15;
            }
        } else if curr_state.do_remove_block {
            if let Some(inter) = curr_state.look_at_block {
                overworld.access().update_block(&inter.0, |data| {
                    data.set(registry.block_empty);
                    *data.raw_light_source_mut() = LightLevel::ZERO;
                });
                check_block(registry.registry(), &mut overworld.access(), &inter.0);

                let mut access = main_state.lock().overworld.access().into_read_only();
                let data = access.get_block(&inter.0).unwrap();

                base::on_light_tick(
                    &inter.0,
                    data,
                    registry.registry(),
                    &mut main_state.lock().overworld.access().into_read_only(),
                    new_actions.builder(),
                );
                base::on_sky_light_tick(
                    &inter.0,
                    data,
                    registry.registry(),
                    &mut main_state.lock().overworld.access().into_read_only(),
                    new_actions.builder(),
                );

                new_block_set_cooldown = 0.15;
            }
        }
    } else {
        new_block_set_cooldown = (curr_state.block_set_cooldown - delta_time).max(0.0);
    }

    if curr_state.change_stream_pos {
        curr_state
            .overworld_orchestrator
            .as_ref()
            .unwrap()
            .lock()
            .set_stream_pos(player_pos);
    }

    let mut new_state = main_state.lock();
    new_state.last_tick_start = start_t;
    new_state.block_set_cooldown = new_block_set_cooldown;

    curr_state.overworld.interface().update_persisted_state(&|state| {
        state.update_player_state(|p_state| {
            p_state.set_satiety(
                (p_state.satiety() - MainApp::SATIETY_IDLE_DECREASE_SPEED * delta_time).max(0.0),
            );

            if p_state.health() > 0.0 && p_state.health() < MainApp::HEALTH_REGEN_THRESHOLD {
                let regen = p_state.satiety() * MainApp::HEALTH_MAX_REGEN_SPEED * delta_time;
                p_state.set_health((p_state.health() + regen).min(MainApp::HEALTH_REGEN_THRESHOLD));
            }
        });
    });
}
