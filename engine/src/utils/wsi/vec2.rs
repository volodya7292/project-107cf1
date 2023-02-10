use crate::utils::wsi::{real_scale_factor, real_window_size};
use nalgebra_glm as glm;
use nalgebra_glm::{TVec2, UVec2, Vec2};
use std::fmt::Debug;
use winit::window::Window;

pub trait WSINumber: Copy + PartialEq + Debug + 'static {}

impl WSINumber for u32 {}
impl WSINumber for f32 {}

#[derive(Copy, Clone, PartialEq)]
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

    pub fn from_winit(raw: (u32, u32), window: &Window) -> Self {
        let logical_win_size = window.inner_size();
        let real_win_size = real_window_size(window);
        let real = glm::vec2(
            raw.0 * real_win_size.width / logical_win_size.width,
            raw.1 * real_win_size.height / logical_win_size.height,
        );
        Self {
            real,
            scale_factor: real_scale_factor(window) as f32,
        }
    }

    pub fn logical(&self) -> Vec2 {
        glm::convert::<_, Vec2>(self.real) / self.scale_factor
    }
}

impl WSIVec2<f32> {
    pub fn from_winit(raw: (f32, f32), window: &Window) -> Self {
        let logical_win_size = window.inner_size();
        let real_win_size = real_window_size(window);
        let real = glm::vec2(
            raw.0 * real_win_size.width as f32 / logical_win_size.width as f32,
            raw.1 * real_win_size.height as f32 / logical_win_size.height as f32,
        );
        WSIVec2 {
            real,
            scale_factor: real_scale_factor(window) as f32,
        }
    }

    pub fn logical(&self) -> Vec2 {
        self.real / self.scale_factor
    }
}
