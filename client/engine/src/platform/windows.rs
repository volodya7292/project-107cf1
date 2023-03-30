use crate::platform::{Platform, PlatformImpl};
use windows_sys::Win32::Foundation::S_OK;
use windows_sys::Win32::UI::HiDpi::{GetDpiForMonitor, MDT_RAW_DPI};
use winit::monitor::MonitorHandle;
use winit::platform::windows::MonitorHandleExtWindows;

impl PlatformImpl for Platform {
    fn get_monitor_dpi(monitor: &MonitorHandle) -> Option<u32> {
        let mut dpi_x = 0;
        let mut dpi_y = 0;
        let result = unsafe { GetDpiForMonitor(monitor.hmonitor(), MDT_RAW_DPI, &mut dpi_x, &mut dpi_y) };
        (result == S_OK).then(|| dpi_x)
    }
}
