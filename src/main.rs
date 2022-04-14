mod game;
mod material_pipelines;
mod physics;
mod scene_component;
#[cfg(test)]
mod tests;
mod utils;

use crate::game::Game;
use engine::utils::noise::ParamNoise;
use engine::Engine;
use nalgebra_glm as glm;
use nalgebra_glm::DVec2;
use noise::Seedable;
use simple_logger::SimpleLogger;

pub const PROGRAM_NAME: &str = "project-107cf1";

fn noise_test() {
    let mut buf = vec![0_u8; 1024 * 1024 * 3];

    let n = noise::SuperSimplex::new().set_seed(0);

    let process = |p: DVec2| -> f64 {
        let v = n.sample(p, 5.0, 1.0, 0.5);

        let d = (glm::distance(&DVec2::new(0.5, 0.5), &p) * 2.0).min(1.0);
        let s = glm::smoothstep(0.0, 1.0, 4.0 * (1.0 - d));

        // y = 0.4 - (0.1 - (x - 0.4)) / 0.1

        v * s
    };

    for x in 0..1024 {
        for y in 0..1024 {
            let i = (y * 1024 + x) * 3;
            let x = x as f64 / 1024.0;
            let y = y as f64 / 1024.0;

            let v = process(DVec2::new(x, y));

            buf[i] = (v * 255.0) as u8;
            buf[i + 1] = buf[i];
            buf[i + 2] = buf[i];
        }
    }

    image::save_buffer("D:/noise_test.png", &buf, 1024, 1024, image::ColorType::Rgb8).unwrap();
    std::process::exit(0);
}

fn parking_lot_deadlock_detection() {
    use parking_lot::deadlock;
    use std::thread;
    use std::time::Duration;

    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(10));

        let deadlocks = deadlock::check_deadlock();
        if deadlocks.is_empty() {
            continue;
        }

        println!("{} deadlocks detected", deadlocks.len());
        for (i, threads) in deadlocks.iter().enumerate() {
            println!("Deadlock #{}", i);
            for t in threads {
                println!("Thread Id {:#?}", t.thread_id());
                println!("{:#?}", t.backtrace());
            }
        }
    });
}

fn main() {
    parking_lot_deadlock_detection();

    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();

    let game = Box::new(Game::init());
    let engine = Engine::init(PROGRAM_NAME, 4, game);

    engine.run();
}
