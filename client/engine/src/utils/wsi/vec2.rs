use crate::utils::wsi::{find_best_video_mode, real_scale_factor};
use common::glm;
use common::glm::{TVec2, Vec2};
use std::fmt::Debug;
use winit::dpi::PhysicalSize;
use winit::window::Window;

pub trait WSINumber: Copy + PartialEq + Debug + 'static {}

impl WSINumber for u32 {}
impl WSINumber for f32 {}

pub struct WSizingInfo {
    monitor_native_size: PhysicalSize<u32>,
    // On macOS this may be larger than native size
    monitor_logical_size: PhysicalSize<u32>,
    scale_factor: f32,
}

impl WSizingInfo {
    pub fn get(window: &Window) -> Self {
        let monitor = window.current_monitor().unwrap();
        let native_mode = find_best_video_mode(&monitor);
        let native_size = native_mode.size();
        let logical_size = monitor.size();

        Self {
            monitor_native_size: native_size,
            monitor_logical_size: logical_size,
            scale_factor: real_scale_factor(window) as f32,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct WSIVec2<T: WSINumber> {
    real: TVec2<T>,
    scale_factor: f32,
}

impl<T: WSINumber> WSIVec2<T> {
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    pub fn real(&self) -> TVec2<T> {
        self.real
    }
}

impl WSIVec2<u32> {
    pub fn from_logical(logical: Vec2, window: &Window) -> WSIVec2<u32> {
        let scale_factor = real_scale_factor(window) as f32;

        let real = logical * scale_factor;
        WSIVec2 {
            real: glm::convert_unchecked(real),
            scale_factor,
        }
    }

    pub fn from_raw(raw: (u32, u32), sizing_info: &WSizingInfo) -> Self {
        let native_size = sizing_info.monitor_native_size;
        let logical_size = sizing_info.monitor_logical_size;

        let real = glm::vec2(
            raw.0 * native_size.width / logical_size.width,
            raw.1 * native_size.height / logical_size.height,
        );

        Self {
            real,
            scale_factor: sizing_info.scale_factor,
        }
    }

    pub fn logical(&self) -> Vec2 {
        glm::convert::<_, Vec2>(self.real) / self.scale_factor
    }
}

impl WSIVec2<f32> {
    pub fn from_raw(raw: (f32, f32), sizing_info: &WSizingInfo) -> Self {
        let native_size = sizing_info.monitor_native_size;
        let logical_size = sizing_info.monitor_logical_size;

        let real = glm::vec2(
            raw.0 * native_size.width as f32 / logical_size.width as f32,
            raw.1 * native_size.height as f32 / logical_size.height as f32,
        );

        WSIVec2 {
            real,
            scale_factor: sizing_info.scale_factor,
        }
    }

    pub fn logical(&self) -> Vec2 {
        self.real / self.scale_factor
    }
}
