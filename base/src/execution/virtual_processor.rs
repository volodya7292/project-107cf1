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
    complete_notify: Arc<Notify>,
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
            let listener = task.complete_notify.notified();
            pool.spawn(move || {
                if task.cancelled.load(MO_RELAXED) {
                    return;
                }
                (task.func)()
            });
            listener.await;
        }
    }

    pub fn spawn<T, F>(&self, task: F) -> VirtualTask<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let result = Arc::new(Mutex::new(None::<T>));
        let completion_notify = Arc::new(Notify::new());

        let closure = {
            let result = Arc::clone(&result);
            let completion_notify = Arc::clone(&completion_notify);
            move || {
                let output = task();
                *result.lock() = Some(output);
                completion_notify.notify_waiters();
            }
        };
        let scheduled_task = ScheduledTask {
            func: Box::new(closure),
            cancelled: Arc::new(AtomicBool::new(false)),
            complete_notify: Arc::clone(&completion_notify),
        };
        let virtual_task = VirtualTask {
            result,
            completion_notify,
            cancelled: Arc::clone(&scheduled_task.cancelled),
        };

        self.sender.send(scheduled_task).ok().unwrap();
        virtual_task
    }
}

pub struct VirtualTask<T> {
    result: Arc<Mutex<Option<T>>>,
    completion_notify: Arc<Notify>,
    cancelled: Arc<AtomicBool>,
}

impl<T> VirtualTask<T> {
    pub fn is_finished(&self) -> bool {
        self.result.lock().is_some()
    }

    pub fn cancel(self) {
        self.cancelled.store(true, MO_RELAXED);
    }

    pub async fn future(self) -> T {
        loop {
            let listener = self.completion_notify.notified();
            if let Some(result) = self.result.lock().take() {
                return result;
            }
            listener.await;
        }
    }
}
