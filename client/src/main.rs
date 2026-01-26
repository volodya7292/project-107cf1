mod client;
mod default_resource_mapping;
mod game;
pub mod proto;
mod rendering;
mod resource_mapping;
#[cfg(test)]
mod tests;

use crate::game::MainApp;
use base::execution::RuntimeGuard;
use common::log;
use engine::Engine;
use simple_logger::SimpleLogger;
use std::thread;

fn parking_lot_deadlock_detection() {
    use common::parking_lot::deadlock;
    use std::time::Duration;

    thread::spawn(move || {
        loop {
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
        }
    });
}

fn init_threads() -> RuntimeGuard {
    let mut available_threads = thread::available_parallelism().unwrap().get();

    let render_threads = (available_threads / 2).min(2);
    available_threads = available_threads.saturating_sub(render_threads).max(1);

    let default_threads = available_threads;

    // available_threads = available_threads.saturating_sub(coroutine_threads).max(1);

    let guard = base::execution::init(default_threads);
    engine::execution::init(render_threads);
    guard
}

fn main() {
    // parking_lot_deadlock_detection();
    let _rt_guard = init_threads();

    SimpleLogger::new()
        .with_level(log::LevelFilter::Debug)
        .init()
        .unwrap();

    let engine = Engine::init(MainApp::create_window, MainApp::init);

    base::execution::spawn_coroutine(async {
        println!("coroutine test!");
    })
    .detach();

    engine.run();
}
