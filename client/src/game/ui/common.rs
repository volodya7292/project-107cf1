pub use crate::game::ui::inventory::game_inventory_overlay;
pub use crate::game::{EngineCtxGameExt, MainApp};
pub use crate::rendering::item_visuals::ItemVisuals;
pub use crate::rendering::ui::backgrounds::game_effects;
pub use crate::rendering::ui::container::{
    background, container, container_props, container_props_init, expander, height_spacer, width_spacer,
};
pub use crate::rendering::ui::fancy_button::fancy_button;
pub use crate::rendering::ui::fancy_text_input::{FancyTextInputProps, fancy_text_input};
pub use crate::rendering::ui::image::reactive::{UIImageProps, ui_image, ui_image_props};
pub use crate::rendering::ui::image::{ImageFitness, ImageSource};
pub use crate::rendering::ui::scrollable_container::scrollable_container;
pub use crate::rendering::ui::text::reactive::{ui_text, ui_text_props};
pub use crate::rendering::ui::{STATE_ENTITY_ID, backgrounds, ui_callbacks};
pub use base::overworld::item::ItemStack;
pub use base::player::{self, NUM_INTENTORY_ITEM_SLOTS};
pub use common::glm::Vec2;
pub use common::make_static_id;
pub use common::types::CmpArc;
pub use engine::ecs::component::simple_text::{StyledString, TextHAlign, TextStyle};
pub use engine::ecs::component::ui::{ClickedCallback, CrossAlign, Padding, Position, Sizing, UILayoutC};
pub use engine::event::WSIKeyboardInput;
pub use engine::module::ui::color::Color;
pub use engine::module::ui::reactive::{ReactiveState, UIReactor, UIScopeContext};
pub use engine::utils::transition::{AnimatedValue, TransitionTarget};
pub use engine::winit::event::ElementState;
pub use engine::winit::keyboard::KeyCode;
pub use engine::{EngineContext, remember_state};
pub use entity_data::EntityId;
pub use std::sync::Arc;

pub type ModalFn = CmpArc<dyn Fn(&mut UIScopeContext) + Send + Sync + 'static>;

pub mod ui_root_states {
    use super::ModalFn;
    use base::overworld::item::ItemStack;
    use engine::module::ui::reactive::StateId;
    use lazy_static::lazy_static;

    lazy_static! {
        pub static ref DEBUG_INFO: StateId<Vec<String>> = "debug_info".into();
        pub static ref DEBUG_INFO_VISIBLE: StateId<bool> = "debug_info_visible".into();
        pub static ref MENU_VISIBLE: StateId<bool> = "menu_visible".into();
        pub static ref ACTIVE_MODAL_VIEWS: StateId<Vec<ModalFn>> = "curr_modal_view".into();
        pub static ref CURR_MENU_TAB: StateId<&'static str> = "curr_menu_tab".into();
        pub static ref IN_GAME_PROCESS: StateId<bool> = "in_game_process".into();
        pub static ref VISION_OBSTRUCTED: StateId<bool> = "vision_obstructed".into();
        pub static ref PLAYER_HEALTH: StateId<f64> = "player_health".into();
        pub static ref PLAYER_SATIETY: StateId<f64> = "player_satiety".into();
        pub static ref WORLD_NAME_LIST: StateId<Vec<String>> = "world_name_list".into();
        pub static ref INVENTORY_VISIBLE: StateId<bool> = "inventory_visible".into();
        pub static ref INVENTORY_ITEMS: StateId<Vec<ItemStack>> = "inventory_items".into();
        pub static ref SELECTED_INVENTORY_ITEM_IDX: StateId<u32> = "selected_inventory_item_idx".into();
    }
}

pub const TAB_TITLE_COLOR: Color = Color::rgb(0.5, 1.8, 0.5);
// const BUTTON_TEXT_COLOR: Color = Color::rgb(3.0, 6.0, 3.0);
pub const BUTTON_TEXT_COLOR: Color = Color::rgb(0.9, 2.0, 0.9);
pub const TEXT_COLOR: Color = Color::grayscale(0.9);
