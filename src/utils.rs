pub fn find_largest_video_mode(monitor: &winit::monitor::MonitorHandle) -> winit::monitor::VideoMode {
    let modes: Vec<_> = monitor.video_modes().collect();
    let mut largest_mode = modes[0].clone();

    for mode in &modes {
        if mode.size().width > largest_mode.size().width {
            largest_mode = mode.clone();
        }
    }

    largest_mode
}

macro_rules! unwrap_option {
    ($to_unwrap: expr, $on_else: expr) => {
        if let Some(v) = $to_unwrap {
            v
        } else {
            $on_else
        }
    };
}

pub(crate) use unwrap_option;
