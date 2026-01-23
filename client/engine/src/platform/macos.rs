use crate::platform::{Platform, PlatformImpl};
use core_graphics::display::{CGDisplay, CGDisplayMode, kDisplayModeNativeFlag};
use std::ptr;
use winit::monitor::MonitorHandle;
use winit::platform::macos::MonitorHandleExtMacOS;

const MM_PER_INCH: f64 = 25.4;

impl PlatformImpl for Platform {
    fn get_monitor_dpi(monitor: &MonitorHandle) -> Option<u32> {
        let display = CGDisplay::new(monitor.native_id());

        let modes = CGDisplayMode::all_display_modes(display.id, ptr::null())?;

        let native_mode = modes.iter().find(|mode| {
            let flags = mode.io_flags();
            flags & kDisplayModeNativeFlag != 0
        })?;

        let screen_size = display.screen_size();
        let width_inches = screen_size.width / MM_PER_INCH;
        let width_pixels = native_mode.width();

        let dpi = (width_pixels as f64 / width_inches).ceil() as u32;

        // Do not destroy display modes because they are referenced by `CGDisplayMode::all_display_modes`
        for mode in modes {
            std::mem::forget(mode);
        }

        Some(dpi)
    }
}
