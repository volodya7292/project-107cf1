use crate::execution::virtual_processor::VirtualProcessor;
use crate::execution::{Task, spawn_coroutine};
use common::tokio;
use common::tokio::time::MissedTickBehavior;
use std::sync::Arc;
use std::time::Duration;

/// A fair timer that tries to execute each tick on a thread pool as precisely as possible.
pub struct IntervalTimer {
    processor: Arc<VirtualProcessor>,
    waiter: Task<()>,
}

impl IntervalTimer {
    pub fn start<F>(period: Duration, processor: VirtualProcessor, on_tick: F) -> Self
    where
        F: Fn() + Send + Sync + Clone + 'static,
    {
        let processor = Arc::new(processor);

        let waiter = {
            let processor = Arc::clone(&processor);
            spawn_coroutine(async move {
                let mut interval = tokio::time::interval(period);
                interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
                loop {
                    interval.tick().await;
                    processor.spawn(on_tick.clone()).future().await;
                }
            })
        };

        Self { processor, waiter }
    }

    /// May cause deadlock if executed in another virtual processor or thread pool.
    pub fn stop_and_join(&self) {
        self.waiter.cancel();
        self.processor.stop_and_join();
    }
}
