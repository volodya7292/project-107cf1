mod client;
mod default_resources;
mod game;
pub mod proto;
mod rendering;
mod resource_mapping;
#[cfg(test)]
mod tests;

use crate::game::Game;
use common::log;
use engine::{utils, Engine};
use simple_logger::SimpleLogger;
use std::thread;
use std::time::{Duration, Instant};

pub const PROGRAM_NAME: &str = "project-107cf1";

#[cfg(target_os = "macos")]
embed_plist::embed_info_plist!("../Info.plist");

fn parking_lot_deadlock_detection() {
    use common::parking_lot::deadlock;
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

fn init_threads() {
    let mut available_threads = thread::available_parallelism().unwrap().get();

    let render_threads = (available_threads / 2).min(2);
    available_threads = available_threads.saturating_sub(render_threads).max(1);

    let default_threads = available_threads;

    let coroutine_threads = 1;
    // available_threads = available_threads.saturating_sub(coroutine_threads).max(1);

    base::execution::init(default_threads, coroutine_threads);
    engine::execution::init(render_threads);
}

fn main() {
    parking_lot_deadlock_detection();
    init_threads();

    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();

    let game = Game::init();
    let engine = Engine::init(game);

    base::execution::spawn_coroutine(async {
        println!("coroutine test!");
    });

    engine.run();
}
