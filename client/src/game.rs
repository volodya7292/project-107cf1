mod ui;

use crate::default_resources::DefaultResourceMapping;
use crate::game::ui::ui_root_states;
use crate::rendering::material_pipelines;
use crate::rendering::material_pipelines::MaterialPipelines;
use crate::rendering::overworld_renderer::OverworldRenderer;
use crate::rendering::ui::image::{ImageAccess, ImageImpl};
use crate::rendering::ui::register_ui_elements;
use crate::rendering::ui::text::UITextImpl;
use approx::AbsDiffEq;
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
use base::overworld::raw_cluster::{BlockDataImpl, LightType, RawCluster};
use base::overworld::{Overworld, OverworldOrchestrator, OverworldParams};
use base::physics::aabb::AABB;
use base::physics::MOTION_EPSILON;
use common::glm;
use common::glm::{DVec3, I64Vec3, Vec2, Vec3};
use common::lrc::{Lrc, LrcExtSized, OwnedRefMut};
use common::parking_lot::Mutex;
use common::rayon::prelude::*;
use common::resource_file::{BufferedResourceReader, ResourceFile};
use common::threading::SafeThreadPool;
use engine::event::{WSIEvent, WSIKeyboardInput};
use engine::module::input::Input;
use engine::module::main_renderer::{camera, MainRenderer, WrapperObject};
use engine::module::scene::{ObjectEntityId, Scene};
use engine::module::text_renderer::{FontSet, TextRenderer};
use engine::module::ui::reactive::UIReactor;
use engine::module::ui::{UIObjectEntityImpl, UIRenderer};
use engine::module::ui_interaction_manager::UIInteractionManager;
use engine::module::{main_renderer, EngineModule};
use engine::utils::wsi::find_best_video_mode;
use engine::winit::event::{MouseButton, VirtualKeyCode};
use engine::winit::event_loop::EventLoop;
use engine::winit::window::{CursorGrabMode, Fullscreen, Window, WindowBuilder};
use engine::{winit, EngineContext};
use entity_data::EntityId;
use std::any::Any;
use std::cell::RefMut;
use std::f32::consts::FRAC_PI_2;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PROGRAM_NAME: &str = "project-107cf1";
const DEF_WINDOW_SIZE: (u32, u32) = (1280, 720);
const PLAYER_CAMERA_OFFSET: DVec3 = DVec3::new(0.0, 0.625, 0.0);

pub trait EngineCtxGameExt {
    fn app(&self) -> OwnedRefMut<dyn EngineModule, MainApp>;
    fn resources(&self) -> Arc<BufferedResourceReader>;
}

impl EngineCtxGameExt for EngineContext<'_> {
    fn app(&self) -> OwnedRefMut<dyn EngineModule, MainApp> {
        self.module_mut::<MainApp>()
    }

    fn resources(&self) -> Arc<BufferedResourceReader> {
        let scene = self.module::<Scene>();
        Arc::clone(&scene.resource::<Arc<BufferedResourceReader>>())
    }
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

    cursor_grab: bool,
    root_entity: EntityId,
    ui_reactor: Lrc<UIReactor>,
}

impl MainApp {
    const MOUSE_SENSITIVITY: f64 = 0.2;

    pub fn init(ctx: &EngineContext) {
        ctx.register_module(Scene::new());

        let root_entity = *ctx
            .module_mut::<Scene>()
            .add_object(None, WrapperObject::new())
            .unwrap();

        let renderer = MainRenderer::new(
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
            root_entity,
            ctx,
        );
        ctx.register_module(renderer);

        ctx.register_module(TextRenderer::new(ctx));
        ctx.register_module(UIRenderer::new(ctx, root_entity));
        ctx.register_module(UIInteractionManager::new(ctx));
        ctx.register_module(Input::new());

        register_ui_elements(ctx);

        let resources = BufferedResourceReader::new(ResourceFile::open("resources").unwrap());

        let main_registry = MainRegistry::init();
        let res_map = DefaultResourceMapping::init(&main_registry, resources.file());

        let mat_pipelines;
        {
            mat_pipelines = material_pipelines::create(resources.file(), &ctx);

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

        let ui_reactor = { UIReactor::new(move |ctx| ui::ui_root(ctx, ui_root_element)) };

        let game = MainApp {
            resources: Arc::clone(&resources),
            main_registry: Arc::clone(&main_registry),
            res_map,
            default_queue: default_queue().unwrap(),
            cursor_rel: Default::default(),
            tick_timer: None,
            material_pipelines: mat_pipelines,
            game_state: None,
            cursor_grab: true,
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

    pub fn is_in_game(&self) -> bool {
        self.game_state.is_some()
    }

    pub fn is_main_menu_visible(&self, _: &EngineContext) -> bool {
        let mut ui_reactor = self.ui_reactor.borrow_mut();
        ui_reactor
            .root_state::<bool>(ui_root_states::MENU_VISIBLE.to_string())
            .map_or(false, |v| *v.value())
    }

    pub fn show_main_menu(&mut self, ctx: &EngineContext, visible: bool) {
        let mut ui_reactor = self.ui_reactor.borrow_mut();
        ui_reactor
            .state::<bool>(
                UIReactor::ROOT_SCOPE_ID.to_string(),
                ui_root_states::MENU_VISIBLE.to_string(),
            )
            .unwrap()
            .update(visible);
        drop(ui_reactor);

        self.grab_cursor(&ctx.window(), !visible, ctx);
    }

    pub fn make_world_path(overworld_name: &str) -> PathBuf {
        Self::worlds_dir().join(overworld_name)
    }

    pub fn create_overworld(&mut self, overworld_name: &str, seed_str: &str) {
        let world_path = Self::make_world_path(overworld_name);

        let seed_digest = common::ring::digest::digest(&common::ring::digest::SHA256, seed_str.as_bytes());
        let seed = u64::from_le_bytes((seed_digest.as_ref()[..8]).try_into().unwrap());

        LocalOverworldInterface::create_overworld(world_path, OverworldParams { seed, tick_count: 0 });
    }

    pub fn delete_overworld(&mut self, overworld_name: &str) {
        let world_path = Self::make_world_path(overworld_name);
        let _ = fs::remove_dir_all(world_path);
    }

    pub fn start_game_process(&mut self, ctx: &EngineContext, overworld_name: &str) {
        let world_path = Self::make_world_path(overworld_name);

        let interface = LocalOverworldInterface::new(world_path, &self.main_registry);
        let overworld = Overworld::new(&self.main_registry, Arc::new(interface));

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
            tick_count: 0,
            overworld,
            overworld_orchestrator,
            overworld_renderer: Arc::clone(&overworld_renderer),
            res_map: Arc::clone(&self.res_map),
            player_pos,
            player_orientation: Default::default(),
            player_direction: Default::default(),
            player_aabb: AABB::from_size(DVec3::new(0.6, 1.75, 0.6)),
            fall_time: 0.0,
            curr_jump_force: 0.0,
            look_at_block: None,
            block_set_cooldown: 0.0,
            do_set_block: false,
            curr_block: self.main_registry.block_test.into_any(),
            set_water: false,
            change_stream_pos: true,
            player_collision_enabled: false,
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

        self.show_main_menu(ctx, false);
        self.grab_cursor(&*ctx.window(), true, ctx);
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
}

// TODO: Make Application this a module

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

        let Some(main_state) = self.game_state.as_ref() else {
            return;
        };

        let mut renderer = ctx.module_mut::<MainRenderer>();
        let input = ctx.module::<Input>();

        let kb = input.keyboard();
        let mut curr_state = main_state.lock();
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
            let cursor_offset = self.cursor_rel * (Self::MOUSE_SENSITIVITY * delta_time) as f32;

            let mut rotation = *camera.rotation();
            rotation.x = (rotation.x + cursor_offset.y).clamp(-FRAC_PI_2, FRAC_PI_2);
            rotation.y += cursor_offset.x;
            // rotation.y += delta_time as f32;

            camera.set_rotation(rotation);
            camera.set_position(curr_state.player_pos + PLAYER_CAMERA_OFFSET);
            self.cursor_rel = Vec2::new(0.0, 0.0);

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

        let overworld_renderer = Arc::clone(&curr_state.overworld_renderer);

        drop(curr_state);
        drop(renderer);
        drop(input);

        overworld_renderer
            .lock()
            .update_scene(&mut ctx.module_mut::<Scene>());
    }

    fn on_wsi_event(&mut self, main_window: &Window, event: &WSIEvent, ctx: &EngineContext) {
        use engine::winit::event::ElementState;

        if !self.is_main_menu_visible(ctx) && self.game_state.is_some() {
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
                            self.show_main_menu(ctx, !self.is_main_menu_visible(ctx));
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
    overworld_orchestrator: Arc<Mutex<OverworldOrchestrator>>,
    overworld_renderer: Arc<Mutex<OverworldRenderer>>,
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

fn on_tick(main_state: Arc<Mutex<GameProcessState>>, overworld_renderer: Arc<Mutex<OverworldRenderer>>) {
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

fn player_on_update(main_state: &Arc<Mutex<GameProcessState>>, new_actions: &mut OverworldActionsStorage) {
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
