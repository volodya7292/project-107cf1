use std::slice;

use vk_engine::Format;

fn main() {
    simple_logger::init().unwrap();

    let mut windows = windows::Entry::new().unwrap();
    let mut window = windows.create_window(500, 500, "GOVNO").unwrap();

    let windows_extensions = windows.get_required_vk_instance_extensions().unwrap();
    let vke = vk_engine::Entry::new().unwrap();
    let mut instance = vke
        .create_instance("GOVNO!", windows_extensions.iter().map(String::as_str).collect())
        .unwrap();

    //windows.set_fullscreen(&mut window, true);

    let surface = instance.create_surface(&window).unwrap();

    let adapters = instance.enumerate_adapters(surface).unwrap();
    let device = instance.create_device(&adapters[0]).unwrap();

    instance.govno();

    //device.create_buffer::<u32>(32);
    //window.create_vk_surface(&instance).unwrap();

    windows.main_loop(slice::from_ref(&&window));

    instance.destroy_surface(surface);
}
