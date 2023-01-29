#[cfg(target_os = "macos")]
mod macos;
#[cfg(target_os = "windows")]
mod windows;

pub struct Platform;

pub trait PlatformImpl {
    fn get_monitor_dpi(monitor: &winit::monitor::MonitorHandle) -> Option<u32>;
}

pub trait EngineMonitorExt {
    fn dpi(&self) -> Option<u32>;
}

impl EngineMonitorExt for winit::monitor::MonitorHandle {
    fn dpi(&self) -> Option<u32> {
        Platform::get_monitor_dpi(self)
    }
}
