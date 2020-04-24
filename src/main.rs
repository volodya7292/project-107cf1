use std::slice;

fn main() {
    simple_logger::init().unwrap();

    let mut windows = windows::Entry::new().unwrap();
    let mut window = windows.create_window(500, 500, "GOVNO").unwrap();

    //windows.set_fullscreen(&mut window, true);

    windows.main_loop(slice::from_ref(&window));
}
