use crate::execution::{spawn_coroutine, Task};
use common::event_listener::Event;
use common::MO_RELAXED;
use std::future::Future;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

#[derive(Default)]
pub struct Timeline {
    counter: AtomicU64,
    event: Event,
}

impl Timeline {
    pub fn new() -> Self {
        Default::default()
    }
}

#[derive(Clone)]
pub struct TaskSemaphore {
    timeline: Arc<Timeline>,
    wait_value: u64,
}

impl TaskSemaphore {
    /// Waits until the task is completed or cancelled.
    pub async fn wait(&self) {
        while self.timeline.counter.load(MO_RELAXED) < self.wait_value {
            self.timeline.event.listen().await;
        }
    }
}

pub struct RecurringTask {
    inner: Option<Task<()>>,
    timeline: Arc<Timeline>,
    last_signal_value: u64,
}

impl RecurringTask {
    pub fn new() -> Self {
        Self {
            inner: None,
            timeline: Default::default(),
            last_signal_value: 0,
        }
    }

    /// Returns a semaphore that will be signalled after some subsequent scheduled task completes.
    pub fn next_signal_waiter(&self) -> TaskSemaphore {
        TaskSemaphore {
            timeline: Arc::clone(&self.timeline),
            wait_value: self.last_signal_value + 1,
        }
    }

    pub fn reschedule<Fut, F>(&mut self, f: F)
    where
        Fut: Future<Output = ()> + Send,
        F: FnOnce() -> Fut + Send + 'static,
    {
        let mut inner = self.inner.take();
        let timeline = Arc::clone(&self.timeline);

        self.last_signal_value += 1;
        let signal_value = self.last_signal_value;

        let inner = spawn_coroutine(async move {
            if let Some(task) = inner.take() {
                task.cancel().await;
            }
            f().await;
            timeline.counter.store(signal_value, MO_RELAXED);
            timeline.event.notify(usize::MAX);
        });

        self.inner = Some(inner);
    }

    pub fn cancel(&mut self) {
        if let Some(task) = self.inner.take() {
            let _ = task.cancel();
        }
        self.timeline.counter.store(self.last_signal_value, MO_RELAXED);
        self.timeline.event.notify(usize::MAX);
    }
}

impl Drop for RecurringTask {
    fn drop(&mut self) {
        self.cancel();
    }
}
