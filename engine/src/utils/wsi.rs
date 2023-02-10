pub mod vec2;

/// Window System Integration
use crate::platform::EngineMonitorExt;
use crate::utils;
use crate::utils::wsi::vec2::WSIVec2;
use nalgebra_glm as glm;
use nalgebra_glm::{DVec2, IVec2, TVec2, UVec2, Vec2};
use std::fmt::Debug;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::window::Window;

pub type WSIPosition<T> = WSIVec2<T>;
pub type WSISize<T> = WSIVec2<T>;

/// `monitor_scale_factor()` for monitor with this DPI is equal to 1.
pub const DEFAULT_DPI: u32 = 109;

pub fn find_best_video_mode(monitor: &winit::monitor::MonitorHandle) -> winit::monitor::VideoMode {
    let curr_refresh_rate = monitor.refresh_rate_millihertz().unwrap();

    monitor
        .video_modes()
        .max_by(|a, b| {
            let a_width = a.size().width;
            let b_width = b.size().width;
            let a_fps_diff = a.refresh_rate_millihertz().abs_diff(curr_refresh_rate);
            let b_fsp_diff = b.refresh_rate_millihertz().abs_diff(curr_refresh_rate);

            a_width.cmp(&b_width).then(a_fps_diff.cmp(&b_fsp_diff).reverse())
        })
        .unwrap()
}

/// Calculates best UI scale factor for the specified window depending on corresponding monitor's DPI.
pub fn real_scale_factor(window: &Window) -> f64 {
    let monitor = window.current_monitor().unwrap();
    let native_mode = find_best_video_mode(&monitor);

    let dpi = monitor.dpi().unwrap_or_else(|| {
        log::warn!("Failed to get monitor DPI!");
        DEFAULT_DPI
    });

    dpi as f64 / DEFAULT_DPI as f64
}

/// Calculates best UI scale factor relative `window` size depending on corresponding monitor's DPI.
pub fn real_window_size(window: &Window) -> PhysicalSize<u32> {
    let monitor = window.current_monitor().unwrap();
    let native_mode = find_best_video_mode(&monitor);
    let native_size = native_mode.size();
    let logical_size = monitor.size(); // On macOS this may be larger than native width

    let window_size = window.inner_size();
    let real_window_width = window_size.width * native_size.width / logical_size.width;
    let real_window_height = window_size.height * native_size.height / logical_size.height;

    PhysicalSize::new(real_window_width, real_window_height)
}
