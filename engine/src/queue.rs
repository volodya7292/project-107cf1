use crate::{utils, SafeThreadPool, TaskPriority};
use lazy_static::lazy_static;
use std::thread;
use utils::HashMap;

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
                TaskPriority::Coroutine,
                SafeThreadPool::new(coroutine_threads, TaskPriority::Coroutine).unwrap(),
            ),
            (
                TaskPriority::Intensive,
                SafeThreadPool::new(intensive_threads, TaskPriority::Intensive).unwrap(),
            ),
        ]
        .into_iter()
        .collect()
    };
}

pub fn realtime_queue() -> &'static SafeThreadPool {
    &QUEUES[&TaskPriority::Realtime]
}

pub fn coroutine_queue() -> &'static SafeThreadPool {
    &QUEUES[&TaskPriority::Coroutine]
}

pub fn intensive_queue() -> &'static SafeThreadPool {
    &QUEUES[&TaskPriority::Intensive]
}
