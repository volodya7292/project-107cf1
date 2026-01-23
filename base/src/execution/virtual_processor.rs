use crate::execution::{Task, spawn_coroutine};
use common::MO_RELAXED;
use common::parking_lot::Mutex;
use common::threading::SafeThreadPool;
use common::tokio::sync::mpsc::error::TryRecvError;
use common::tokio::sync::{Notify, mpsc};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

struct ScheduledTask {
    func: Box<dyn FnOnce() + Send + 'static>,
    complete_notify: Arc<Notify>,
}

/// A software processor that maps to OS threads.
/// Many such processors allow executing their tasks concurrently
/// instead of waiting for FIFO queue in case of threadpool.
pub struct VirtualProcessor {
    worker: Mutex<Option<Task<()>>>,
    sender: Mutex<Option<mpsc::UnboundedSender<ScheduledTask>>>,
}

impl VirtualProcessor {
    pub fn new(pool: &Arc<SafeThreadPool>) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let worker = spawn_coroutine(VirtualProcessor::worker_fn(Arc::clone(pool), receiver));
        Self {
            worker: Mutex::new(Some(worker)),
            sender: Mutex::new(Some(sender)),
        }
    }

    /// Detaches the virtual processor so the tasks can continue executing in background.
    pub fn detach(self) {
        self.worker.lock().take().unwrap().detach();
    }

    async fn worker_fn(pool: Arc<SafeThreadPool>, mut receiver: mpsc::UnboundedReceiver<ScheduledTask>) {
        loop {
            let n_parallel_tasks = pool.current_num_threads();
            let mut notifiers = Vec::with_capacity(n_parallel_tasks);
            let mut tasks = Vec::with_capacity(n_parallel_tasks);
            let mut do_stop = false;

            for i in 0..n_parallel_tasks {
                let task = if i == 0 {
                    let Some(task) = receiver.recv().await else {
                        do_stop = true;
                        break;
                    };
                    task
                } else {
                    match receiver.try_recv() {
                        Ok(task) => task,
                        Err(TryRecvError::Empty) => {
                            break;
                        }
                        Err(TryRecvError::Disconnected) => {
                            do_stop = true;
                            break;
                        }
                    }
                };
                notifiers.push(task.complete_notify);
                tasks.push(task.func);
            }

            let listeners: Vec<_> = notifiers.iter().map(|notify| notify.notified()).collect();

            for task_fn in tasks.into_iter() {
                pool.spawn(task_fn);
            }

            for listener in listeners {
                listener.await;
            }

            if do_stop {
                break;
            }
        }
    }

    pub fn spawn<T, F>(&self, task: F) -> VirtualTask<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let result = Arc::new(Mutex::new(None::<T>));
        let completion_notify = Arc::new(Notify::new());
        let cancelled = Arc::new(AtomicBool::new(false));

        let closure = {
            let result = Arc::clone(&result);
            let completion_notify = Arc::clone(&completion_notify);
            let cancelled = Arc::clone(&cancelled);
            move || {
                if cancelled.load(MO_RELAXED) {
                    completion_notify.notify_waiters();
                    return;
                }
                let output = task();
                *result.lock() = Some(output);
                completion_notify.notify_waiters();
            }
        };
        let scheduled_task = ScheduledTask {
            func: Box::new(closure),
            complete_notify: Arc::clone(&completion_notify),
        };
        let virtual_task = VirtualTask {
            result,
            completion_notify,
            cancelled,
        };

        if let Some(sender) = &*self.sender.lock() {
            sender.send(scheduled_task).ok().unwrap();
        }
        virtual_task
    }

    fn disconnect_channel(&self) {
        *self.sender.lock() = None;
    }

    /// After joining the virtual processor won't execute any tasks.
    /// May cause a deadlock if executed in another virtual processor or thread pool.
    pub fn stop_and_join(&self) {
        // Cancel the remaining tasks by disconnecting the channel
        self.disconnect_channel();

        if let Some(worker) = self.worker.lock().take() {
            worker.wait();
        } else {
            panic!("virtual processor has already been joined");
        }
    }
}

impl Drop for VirtualProcessor {
    fn drop(&mut self) {
        self.disconnect_channel();
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

    /// Returns result if the task has been completed, `None` otherwise.
    pub fn get_result(self) -> Option<T> {
        self.result.lock().take()
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
