use std::future::Future;
use std::sync::Arc;

use futures_lite::future;
use once_cell::sync::OnceCell;
use rayon::ThreadPool;

use crate::utils::threading;
use crate::utils::threading::{SafeThreadPool, TaskPriority};

static DEFAULT_THREAD_POOL: OnceCell<SafeThreadPool> = OnceCell::new();
static COROUTINE_EXECUTOR: OnceCell<Arc<async_executor::Executor<'static>>> = OnceCell::new();

pub fn init(n_default_threads: usize, n_coroutine_threads: usize) {
    let default_thread_pool = SafeThreadPool::new(n_default_threads, TaskPriority::Default).unwrap();
    let coroutine_executor = Arc::new(async_executor::Executor::new());

    for _ in 0..n_coroutine_threads {
        let executor = Arc::clone(&coroutine_executor);
        threading::spawn_thread(
            move || {
                future::block_on(executor.run(future::pending::<()>()));
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

pub fn spawn_coroutine<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) {
    COROUTINE_EXECUTOR.get().unwrap().spawn(future).detach();
}
