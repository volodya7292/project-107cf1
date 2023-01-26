#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

pub struct Platform;

pub trait PlatformImpl {
    fn get_monitor_dpi(monitor: &winit::monitor::MonitorHandle) -> Option<u32>;
}
