use std::mem;
use std::ops::Deref;
use std::thread::JoinHandle;

/// Safe thread pool that joins all threads on drop.
pub struct SafeThreadPool {
    inner: Option<rayon::ThreadPool>,
    threads: Vec<JoinHandle<()>>,
}

impl SafeThreadPool {
    pub fn new(num_threads: usize) -> Result<SafeThreadPool, rayon::ThreadPoolBuildError> {
        let mut threads = vec![];
        let inner = Some(
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .spawn_handler(|thread| {
                    threads.push(std::thread::spawn(|| thread.run()));
                    Ok(())
                })
                .build()?,
        );

        Ok(SafeThreadPool { inner, threads })
    }
}

impl Deref for SafeThreadPool {
    type Target = rayon::ThreadPool;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl Drop for SafeThreadPool {
    fn drop(&mut self) {
        drop(self.inner.take());
        for thr in mem::take(&mut self.threads) {
            thr.join().unwrap();
        }
    }
}
