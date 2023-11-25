use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

#[derive(Debug, Clone)]
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

    pub fn alloc<T>(&self, count: usize) -> Result<AllocHandle, crate::Error> {
        let bytes = count * std::mem::size_of::<T>();
        let result = self.inner.bytes_left.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |bytes_left| bytes_left.checked_sub(bytes),
        );

        match result {
            Ok(prev) => {
                tracing::trace!(bytes, left = prev - bytes, "Created allocation handle");
                Ok(AllocHandle {
                    bytes,
                    inner: Arc::clone(&self.inner),
                })
            }
            Err(left) => {
                tracing::trace!(bytes, left, "Allocation failed");
                Err(crate::Error::OutOfMemory(bytes))
            }
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
        let bytes = self.bytes;
        let prev = self.inner.bytes_left.fetch_add(bytes, Ordering::Relaxed);
        tracing::trace!(bytes, left = prev + bytes, "Released allocation handle");
        self.bytes = 0;
    }
}

impl AllocHandle {
    pub fn tracker(&self) -> AllocTracker {
        AllocTracker {
            inner: Arc::clone(&self.inner),
        }
    }
}
