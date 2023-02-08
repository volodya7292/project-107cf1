use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crate::utils;
use crate::utils::MO_RELAXED;

pub struct IntervalTimer {
    interval: Duration,
    running: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}
impl IntervalTimer {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            running: Arc::new(Default::default()),
            thread: None,
        }
    }

    pub fn start<F: Fn() + Send + Sync + 'static>(&mut self, on_tick: F) {
        if self.thread.is_some() {
            return;
        }
        self.running.store(true, MO_RELAXED);

        let handle = {
            let running = Arc::clone(&self.running);
            let interval = self.interval;

            thread::spawn(move || loop {
                if !running.load(MO_RELAXED) {
                    break;
                }

                let t0 = Instant::now();
                on_tick();
                let t1 = Instant::now();
                let dt = t1 - t0;

                if dt < interval {
                    let remainder = interval - dt;
                    utils::high_precision_sleep(remainder, Duration::from_micros(100));
                }
            })
        };

        self.thread = Some(handle);
    }

    pub fn stop(&mut self) {
        self.running.store(false, MO_RELAXED);

        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

impl Drop for IntervalTimer {
    fn drop(&mut self) {
        self.stop();
    }
}
