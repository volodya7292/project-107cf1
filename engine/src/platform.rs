use core_video_sys::{
    kCVReturnSuccess, kCVTimeIsIndefinite, CVDisplayLinkCreateWithCGDisplay,
    CVDisplayLinkGetNominalOutputVideoRefreshPeriod, CVDisplayLinkRelease,
};
use winit::platform::macos::MonitorHandleExtMacOS;
use winit::window::Window;

#[cfg(target_os = "macos")]
pub fn current_refresh_rate(window: &Window) -> u32 {
    let curr_display = core_graphics::display::CGDisplay::new(window.current_monitor().unwrap().native_id());
    let cg_refresh_rate = curr_display.display_mode().unwrap().refresh_rate();

    if cg_refresh_rate > 0.0 {
        return cg_refresh_rate as u32;
    }

    let cv_refresh_rate = unsafe {
        let mut display_link = std::ptr::null_mut();
        assert_eq!(
            CVDisplayLinkCreateWithCGDisplay(curr_display.id, &mut display_link),
            kCVReturnSuccess
        );

        let time = CVDisplayLinkGetNominalOutputVideoRefreshPeriod(display_link);
        CVDisplayLinkRelease(display_link);
        assert_eq!(time.flags & kCVTimeIsIndefinite, 0);

        time.timeScale as i64 / time.timeValue
    };

    return cv_refresh_rate as u32;
}
