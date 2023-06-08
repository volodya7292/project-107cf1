use common::event_listener::Event;
use common::futures_lite::future;
use common::parking_lot::Mutex;
use common::threading::{SafeThreadPool, TaskPriority};
use common::{async_executor, threading};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

static DEFAULT_THREAD_POOL: OnceLock<SafeThreadPool> = OnceLock::new();
static COROUTINE_EXECUTOR: OnceLock<Arc<async_executor::Executor<'static>>> = OnceLock::new();

pub struct Task<T>(async_executor::Task<T>);

impl<T> Task<T> {
    pub fn detach(self) {
        self.0.detach()
    }

    pub fn is_finished(&self) -> bool {
        self.0.is_finished()
    }

    pub async fn cancel(self) -> Option<T> {
        self.0.cancel().await
    }
}

impl<T> Future for Task<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        async_executor::Task::poll(Pin::new(&mut self.0), cx)
    }
}

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

fn _spawn_blocking_task<T, F>(f: F, fifo: bool) -> Task<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    let sync_pair = Arc::new((Mutex::new(None::<T>), Event::new()));
    let queue = default_queue();

    {
        let sync_pair = Arc::clone(&sync_pair);
        let closure = move || {
            let result = f();
            *sync_pair.0.lock() = Some(result);
            sync_pair.1.notify(1);
        };
        if fifo {
            queue.spawn_fifo(closure);
        } else {
            queue.spawn(closure);
        }
    }

    Task(COROUTINE_EXECUTOR.get().unwrap().spawn(async move {
        loop {
            let result = sync_pair.0.lock().take();

            if let Some(result) = result {
                return result;
            } else {
                sync_pair.1.listen().await;
            }
        }
    }))
}

pub fn spawn_blocking_task<T, F>(f: F) -> Task<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    _spawn_blocking_task(f, false)
}

pub fn spawn_blocking_task_fifo<T, F>(f: F) -> Task<T>
where
    T: Send + 'static,
    F: FnOnce() -> T + Send + 'static,
{
    _spawn_blocking_task(f, true)
}

pub fn spawn_coroutine<T: Send + 'static>(future: impl Future<Output = T> + Send + 'static) -> Task<T> {
    Task(COROUTINE_EXECUTOR.get().unwrap().spawn(future))
}
