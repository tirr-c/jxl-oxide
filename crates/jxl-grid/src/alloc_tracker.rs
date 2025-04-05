use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Allocation tracker with total memory limit.
#[derive(Debug, Clone)]
pub struct AllocTracker {
    inner: Arc<AllocTrackerInner>,
}

#[derive(Debug)]
struct AllocTrackerInner {
    bytes_left: AtomicUsize,
}

impl AllocTracker {
    /// Creates a memory allocation tracker with allowed allocation limit.
    pub fn with_limit(bytes_left: usize) -> Self {
        Self {
            inner: Arc::new(AllocTrackerInner {
                bytes_left: AtomicUsize::new(bytes_left),
            }),
        }
    }

    /// Records an allocation of `count` number of `T`, and returns handle of the record.
    ///
    /// Returns an error if the allocation exceeds the current limit.
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

    /// Expands the current limit by `by_bytes` bytes.
    pub fn expand_limit(&self, by_bytes: usize) {
        self.inner.bytes_left.fetch_add(by_bytes, Ordering::Relaxed);
    }

    /// Shrinks the current limit by `by_bytes` bytes.
    ///
    /// Returns an error if the total amount of current allocation doesn't allow shrinking the
    /// limit.
    pub fn shrink_limit(&self, by_bytes: usize) -> Result<(), crate::Error> {
        let result = self.inner.bytes_left.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |bytes_left| bytes_left.checked_sub(by_bytes),
        );

        if result.is_ok() {
            Ok(())
        } else {
            Err(crate::Error::OutOfMemory(by_bytes))
        }
    }
}

/// Allocation handle.
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
    /// Returns the tracker the handle belongs to.
    pub fn tracker(&self) -> AllocTracker {
        AllocTracker {
            inner: Arc::clone(&self.inner),
        }
    }
}
