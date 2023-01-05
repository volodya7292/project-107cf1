use std::future::Future;
use std::sync::Arc;
use std::thread;

use futures_lite::future;
use lazy_static::lazy_static;

use core::utils::threading;
use core::utils::HashMap;

use crate::{SafeThreadPool, TaskPriority};

lazy_static! {
    static ref QUEUES: HashMap<TaskPriority, SafeThreadPool> = {
        let mut available_threads = thread::available_parallelism().unwrap().get();

        let render_threads = (available_threads / 2).min(3);
        available_threads = available_threads.saturating_sub(render_threads).max(1);

        let coroutine_threads = 1;
        available_threads = available_threads.saturating_sub(coroutine_threads).max(1);

        let intensive_threads = available_threads;

        [
            (
                TaskPriority::Realtime,
                SafeThreadPool::new(render_threads, TaskPriority::Realtime).unwrap(),
            ),
            (
                TaskPriority::Default,
                SafeThreadPool::new(intensive_threads, TaskPriority::Default).unwrap(),
            ),
        ]
        .into_iter()
        .collect()
    };
    static ref COROUTINE_EXECUTOR: Arc<async_executor::Executor<'static>> = {
        let executor = Arc::new(async_executor::Executor::new());
        {
            let executor = Arc::clone(&executor);
            threading::spawn_thread(
                move || {
                    future::block_on(executor.run(future::pending::<()>()));
                },
                TaskPriority::Default,
            );
        }
        executor
    };
}

pub fn realtime_queue() -> &'static SafeThreadPool {
    &QUEUES[&TaskPriority::Realtime]
}

pub fn default_queue() -> &'static SafeThreadPool {
    &QUEUES[&TaskPriority::Default]
}

pub fn spawn_coroutine<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) {
    COROUTINE_EXECUTOR.spawn(future).detach();
}
