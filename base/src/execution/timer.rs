use crate::execution::virtual_processor::VirtualProcessor;
use crate::execution::{spawn_coroutine, Task};
use common::tokio;
use common::tokio::time::MissedTickBehavior;
use std::time::Duration;

pub struct IntervalTimer {
    _waiter: Task<()>,
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

        Self { _waiter: waiter }
    }
}
