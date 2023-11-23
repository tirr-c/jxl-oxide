use std::sync::{atomic::{AtomicUsize, Ordering}, Arc};

#[derive(Debug)]
pub struct AllocTracker {
    inner: Arc<AllocTrackerInner>,
}

#[derive(Debug)]
struct AllocTrackerInner {
    bytes_left: AtomicUsize,
}

impl AllocTracker {
    pub fn with_limit(bytes_left: usize) -> Self {
        Self {
            inner: Arc::new(AllocTrackerInner {
                bytes_left: AtomicUsize::new(bytes_left),
            }),
        }
    }

    pub fn alloc(&self, bytes: usize) -> Result<AllocHandle, crate::Error> {
        let result = self.inner.bytes_left.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |bytes_left| bytes_left.checked_sub(bytes),
        );

        if result.is_ok() {
            Ok(AllocHandle { bytes, inner: Arc::clone(&self.inner) })
        } else {
            Err(crate::Error::OutOfMemory(bytes))
        }
    }
}

#[derive(Debug)]
pub struct AllocHandle {
    bytes: usize,
    inner: Arc<AllocTrackerInner>,
}

impl Drop for AllocHandle {
    fn drop(&mut self) {
        self.inner.bytes_left.fetch_add(self.bytes, Ordering::Relaxed);
        self.bytes = 0;
    }
}

impl AllocHandle {
    pub fn tracker(&self) -> AllocTracker {
        AllocTracker { inner: Arc::clone(&self.inner) }
    }
}
