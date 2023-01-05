use std::time::{Duration, Instant};

use simple_logger::SimpleLogger;

use engine::{utils, Engine};

use crate::game::Game;

mod client;
mod default_resources;
mod game;
pub mod proto;
mod rendering;
mod resource_mapping;
#[cfg(test)]
mod tests;

pub const PROGRAM_NAME: &str = "project-107cf1";

fn parking_lot_deadlock_detection() {
    use parking_lot::deadlock;
    use std::thread;
    use std::time::Duration;

    thread::spawn(move || loop {
        thread::sleep(Duration::from_secs(5));

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

    engine::queue::spawn_coroutine(async {
        println!("govno");
    });

    engine.run();
}
