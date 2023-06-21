pub mod timer;
pub mod virtual_processor;

use common::futures_lite::{future, FutureExt};
use common::parking_lot::Mutex;
use common::threading::{SafeThreadPool, TaskPriority};
use common::{threading, tokio};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

static DEFAULT_THREAD_POOL: OnceLock<SafeThreadPool> = OnceLock::new();
static COROUTINE_EXECUTOR: OnceLock<Arc<tokio::runtime::Runtime>> = OnceLock::new();

pub struct Task<T>(Option<tokio::task::JoinHandle<T>>);

impl<T> Task<T> {
    pub fn is_finished(&self) -> bool {
        self.0.as_ref().unwrap().is_finished()
    }

    pub fn detach(mut self) {
        self.0 = None;
    }

    pub async fn cancel(self) {
        self.0.as_ref().unwrap().abort();
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

pub fn init(n_default_threads: usize) {
    let default_thread_pool = SafeThreadPool::new(n_default_threads, TaskPriority::Default).unwrap();
    let coroutine_executor = Arc::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap(),
    );

    // Start coroutine executor
    {
        let rt = Arc::clone(&coroutine_executor);
        threading::spawn_thread(
            move || {
                rt.block_on(future::pending::<()>());
            },
            TaskPriority::Default,
        );
    }

    DEFAULT_THREAD_POOL.set(default_thread_pool).unwrap();
    COROUTINE_EXECUTOR.set(coroutine_executor).unwrap();
}

pub fn default_queue() -> &'static SafeThreadPool {
    DEFAULT_THREAD_POOL.get().unwrap()
}

pub fn spawn_coroutine<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) -> Task<T> {
    Task(Some(COROUTINE_EXECUTOR.get().unwrap().spawn(future)))
}
