mod game;
mod material_pipelines;
mod physics;
pub mod proto;
mod scene_component;
#[cfg(test)]
mod tests;
mod utils;

use crate::game::Game;
use engine::Engine;
use simple_logger::SimpleLogger;

pub const PROGRAM_NAME: &str = "project-107cf1";

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
    // std::process::exit(0);
    // parking_lot_deadlock_detection();

    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();

    let game = Box::new(Game::init());
    let engine = Engine::init(PROGRAM_NAME, 4, game);

    engine.run();
}
