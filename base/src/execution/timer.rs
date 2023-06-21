use crate::execution::virtual_processor::VirtualProcessor;
use crate::execution::{spawn_coroutine, Task};
use crate::MO_RELAXED;
use common::tokio;
use common::tokio::time::MissedTickBehavior;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

pub struct IntervalTimer {
    waiter: Task<()>,
}

impl IntervalTimer {
    pub fn start<F>(period: Duration, processor: VirtualProcessor, on_tick: F) -> Self
    where
        F: Fn() + Send + Sync + Clone + 'static,
    {
        let waiter = spawn_coroutine(async move {
            let mut interval = tokio::time::interval(period);
            interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                processor.spawn(on_tick.clone()).future().await;
            }
        });

        Self { waiter }
    }
}
