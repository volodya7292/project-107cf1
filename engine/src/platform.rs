use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
use winit::window::Window;

pub fn current_refresh_rate(window: &Window) -> u32 {
    let handle = match window.raw_window_handle() {
        RawWindowHandle::AppKit(h) => h.ns_window,
        RawWindowHandle::Win32(h) => h.hwnd,
        _ => unreachable!(),
    };

    let sdl = sdl2::init().unwrap();
    let video = sdl.video().unwrap();

    let raw_sdl_window = unsafe { sdl2::sys::SDL_CreateWindowFrom(handle) };
    let sdl_window = unsafe { sdl2::video::Window::from_ll(video, raw_sdl_window) };

    let mode = sdl_window.display_mode().unwrap();
    mode.refresh_rate as u32
}
