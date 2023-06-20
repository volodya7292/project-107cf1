use crate::execution::{default_queue, spawn_coroutine, Task};
use common::parking_lot::Mutex;
use common::threading::SafeThreadPool;
use common::tokio::sync::futures::Notified;
use common::tokio::sync::{mpsc, Notify};
use common::{tokio, MO_RELAXED};
use std::future::{Future, IntoFuture};
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::task::{Context, Poll};

struct ScheduledTask {
    func: Box<dyn FnOnce() + Send + 'static>,
    cancelled: Arc<AtomicBool>,
}

/// Many such processors allow executing their tasks concurrently on fixed number of real cores.
pub struct VirtualProcessor {
    worker: Task<()>,
    sender: mpsc::UnboundedSender<ScheduledTask>,
}

impl VirtualProcessor {
    pub fn new(pool: &'static SafeThreadPool) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let worker = spawn_coroutine(VirtualProcessor::worker_fn(pool, receiver));
        Self { worker, sender }
    }

    async fn worker_fn(pool: &'static SafeThreadPool, mut receiver: mpsc::UnboundedReceiver<ScheduledTask>) {
        while let Some(task) = receiver.recv().await {
            pool.spawn_fifo(move || {
                if task.cancelled.load(MO_RELAXED) {
                    return;
                }
                (task.func)()
            });
        }
    }

    pub fn spawn<T, F>(&self, task: F) -> VirtualTask<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let sync_pair = Arc::new((Mutex::new(None::<T>), Notify::new()));

        let closure = {
            let sync_pair = Arc::clone(&sync_pair);
            move || {
                let result = task();
                *sync_pair.0.lock() = Some(result);
                sync_pair.1.notify_one();
            }
        };
        let scheduled_task = ScheduledTask {
            func: Box::new(closure),
            cancelled: Arc::new(AtomicBool::new(false)),
        };
        let virtual_task = VirtualTask {
            sync_pair,
            cancelled: Arc::clone(&scheduled_task.cancelled),
        };

        self.sender.send(scheduled_task).ok().unwrap();
        virtual_task
    }
}

pub struct VirtualTask<T> {
    sync_pair: Arc<(Mutex<Option<T>>, Notify)>,
    cancelled: Arc<AtomicBool>,
}

impl<T> VirtualTask<T> {
    pub fn is_finished(&self) -> bool {
        self.sync_pair.0.lock().is_some()
    }

    pub fn cancel(self) {
        self.cancelled.store(true, MO_RELAXED);
    }

    pub async fn future(self) -> T {
        loop {
            if let Some(result) = self.sync_pair.0.lock().take() {
                return result;
            }
            self.sync_pair.1.notified().await;
        }
    }
}
