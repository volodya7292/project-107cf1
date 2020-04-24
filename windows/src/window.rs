use std::sync::mpsc::Receiver;

pub struct Window {
    pub(crate) native: glfw::Window,
    pub(crate) events: Receiver<(f64, glfw::WindowEvent)>,
    pub(crate) pos: (i32, i32),
    pub(crate) size: (u32, u32),
}
