pub mod timer;
pub mod virtual_processor;

use common::futures_lite::FutureExt;
use common::parking_lot::RwLock;
use common::threading::{SafeThreadPool, TaskPriority};
use common::tokio::sync::Notify;
use common::{threading, tokio};
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

static DEFAULT_THREAD_POOL: RwLock<Option<Arc<SafeThreadPool>>> = RwLock::new(None);
static COROUTINE_EXECUTOR: RwLock<Option<Arc<tokio::runtime::Runtime>>> = RwLock::new(None);
static COROUTINE_STOP_NOTIFY: RwLock<Option<Arc<Notify>>> = RwLock::new(None);

pub struct Task<T>(Option<tokio::task::JoinHandle<T>>);

impl<T> Task<T> {
    pub fn is_finished(&self) -> bool {
        self.0.as_ref().unwrap().is_finished()
    }

    pub fn detach(mut self) {
        self.0 = None;
    }

    pub fn cancel(self) {
        self.0.as_ref().unwrap().abort();
    }

    pub fn wait(mut self) -> Option<T> {
        COROUTINE_EXECUTOR
            .read()
            .as_ref()
            .unwrap()
            .block_on(self.0.take().unwrap())
            .ok()
    }
}

impl<T> Future for Task<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.0.as_mut().unwrap().poll(cx).map(|res| res.unwrap())
    }
}

impl<T> Drop for Task<T> {
    fn drop(&mut self) {
        if let Some(inner) = self.0.take() {
            inner.abort();
        }
    }
}

pub struct RuntimeGuard {
    _private: (),
}

impl Drop for RuntimeGuard {
    fn drop(&mut self) {
        // Stop thread pool
        {
            let pool = DEFAULT_THREAD_POOL.write().take().unwrap();
            while Arc::strong_count(&pool) > 1 {}
            // Dropping must join the spawned tasks
            drop(pool);
        }

        // Stop async executor
        {
            // Signal to stop
            let stop_notify = COROUTINE_STOP_NOTIFY.write();
            stop_notify.as_ref().unwrap().notify_one();

            let executor = COROUTINE_EXECUTOR.write().take().unwrap();
            while Arc::strong_count(&executor) > 1 {}

            // Wait for completion of remaining tasks
            let executor = Arc::into_inner(executor).unwrap();
            executor.shutdown_timeout(Duration::from_secs(10));
        }
    }
}

pub fn init(n_default_threads: usize) -> RuntimeGuard {
    let default_thread_pool = SafeThreadPool::new(n_default_threads, TaskPriority::Default).unwrap();
    let coroutine_executor = Arc::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    );
    let executor_stop_notify = Arc::new(Notify::new());

    // Start coroutine executor
    {
        let rt = Arc::clone(&coroutine_executor);
        let stop_notify = Arc::clone(&executor_stop_notify);
        threading::spawn_thread(
            move || {
                rt.block_on(async { stop_notify.notified().await });
            },
            TaskPriority::Default,
        );
    }

    *DEFAULT_THREAD_POOL.write() = Some(Arc::new(default_thread_pool));
    *COROUTINE_EXECUTOR.write() = Some(coroutine_executor);
    *COROUTINE_STOP_NOTIFY.write() = Some(executor_stop_notify);
    RuntimeGuard { _private: () }
}

pub fn default_queue() -> Option<Arc<SafeThreadPool>> {
    DEFAULT_THREAD_POOL.read().clone()
}

pub fn spawn_coroutine<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) -> Task<T> {
    let executor = COROUTINE_EXECUTOR.read();
    let executor = executor.as_ref().unwrap();
    Task(Some(executor.spawn(future)))
}
