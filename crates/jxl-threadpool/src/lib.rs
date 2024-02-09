//! Internal crate used in jxl-oxide, for abstraction of thread pool.
//!
//! [`JxlThreadPool`] is re-exported by `jxl-oxide`.

/// Thread pool wrapper.
///
/// This struct wraps internal thread pool implementation and provides interfaces to access it. If
/// `rayon` feature is enabled, users can create an actual thread pool backed by Rayon; if not,
/// this struct won't have any multithreading capability, and every spawn operation will just run
/// the given closure in place.
#[derive(Debug, Clone)]
pub struct JxlThreadPool(JxlThreadPoolImpl);

#[derive(Debug, Clone)]
enum JxlThreadPoolImpl {
    #[cfg(feature = "rayon")]
    Rayon(std::sync::Arc<rayon_core::ThreadPool>),
    None,
}

/// Fork-join scope created by thread pool.
#[derive(Debug, Copy, Clone)]
pub struct JxlScope<'r, 'scope>(JxlScopeInner<'r, 'scope>);

#[derive(Debug, Copy, Clone)]
enum JxlScopeInner<'r, 'scope> {
    #[cfg(feature = "rayon")]
    Rayon(&'r rayon_core::Scope<'scope>),
    None(std::marker::PhantomData<&'r &'scope ()>),
}

impl JxlThreadPool {
    /// Creates a "fake" thread pool without any multithreading capability.
    ///
    /// Every spawn operation on this thread poll will just run the closure in current thread.
    pub const fn none() -> Self {
        Self(JxlThreadPoolImpl::None)
    }

    /// Creates a thread pool backed by Rayon [`ThreadPool`][rayon_core::ThreadPool].
    #[cfg(feature = "rayon")]
    pub fn with_rayon_thread_pool(pool: std::sync::Arc<rayon_core::ThreadPool>) -> Self {
        Self(JxlThreadPoolImpl::Rayon(pool))
    }

    /// Creates a thread pool backed by Rayon.
    ///
    /// If `num_threads_requested` is `None` or zero, this method queries available paralleism and
    /// uses it.
    #[cfg(feature = "rayon")]
    pub fn rayon(num_threads_requested: Option<usize>) -> Self {
        let num_threads_requested = num_threads_requested.unwrap_or(0);

        let num_threads = if num_threads_requested == 0 {
            let num_threads = std::thread::available_parallelism();
            match num_threads {
                Ok(num_threads) => num_threads.into(),
                Err(e) => {
                    tracing::warn!(%e, "Failed to query available parallelism; falling back to single-threaded");
                    return Self::none();
                }
            }
        } else {
            num_threads_requested
        };

        let inner = rayon_core::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map(|pool| JxlThreadPoolImpl::Rayon(std::sync::Arc::new(pool)));

        match inner {
            Ok(inner) => {
                tracing::debug!(num_threads, "Initialized Rayon thread pool");
                Self(inner)
            }
            Err(e) => {
                tracing::warn!(%e, "Failed to initialize thread pool; falling back to single-threaded");
                Self::none()
            }
        }
    }

    /// Returns the reference to Rayon thread pool, if exists.
    #[cfg(feature = "rayon")]
    pub fn as_rayon_pool(&self) -> Option<&rayon_core::ThreadPool> {
        match &self.0 {
            JxlThreadPoolImpl::Rayon(pool) => Some(&**pool),
            JxlThreadPoolImpl::None => None,
        }
    }

    /// Returns if the thread pool is capable of multithreading.
    pub fn is_multithreaded(&self) -> bool {
        match self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(_) => true,
            JxlThreadPoolImpl::None => false,
        }
    }
}

impl JxlThreadPool {
    /// Runs the given closure on the thread pool.
    pub fn spawn(&self, op: impl FnOnce() + Send + 'static) {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.spawn(op),
            JxlThreadPoolImpl::None => op(),
        }
    }

    /// Creates a fork-join scope of tasks.
    pub fn scope<'scope, R: Send>(
        &'scope self,
        op: impl for<'r> FnOnce(JxlScope<'r, 'scope>) -> R + Send,
    ) -> R {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.scope(|scope| {
                let scope = JxlScope(JxlScopeInner::Rayon(scope));
                op(scope)
            }),
            JxlThreadPoolImpl::None => op(JxlScope(JxlScopeInner::None(Default::default()))),
        }
    }

    /// Consumes the `Vec`, and runs a job for each element of the `Vec`.
    pub fn for_each_vec<T: Send>(&self, v: Vec<T>, op: impl Fn(T) + Send + Sync) {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.install(|| par_for_each(v, op)),
            JxlThreadPoolImpl::None => v.into_iter().for_each(op),
        }
    }

    /// Consumes the `Vec`, and runs a job for each element of the `Vec`.
    pub fn for_each_vec_with<T: Send, U: Send + Clone>(
        &self,
        v: Vec<T>,
        init: U,
        op: impl Fn(&mut U, T) + Send + Sync,
    ) {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.install(|| par_for_each_with(v, init, op)),
            JxlThreadPoolImpl::None => {
                let mut init = init;
                v.into_iter().for_each(|item| op(&mut init, item))
            }
        }
    }

    /// Runs a job for each element of the mutable slice.
    pub fn for_each_mut_slice<'a, T: Send>(
        &self,
        v: &'a mut [T],
        op: impl Fn(&'a mut T) + Send + Sync,
    ) {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.install(|| par_for_each(v, op)),
            JxlThreadPoolImpl::None => v.iter_mut().for_each(op),
        }
    }

    /// Runs a job for each element of the mutable slice.
    pub fn for_each_mut_slice_with<'a, T: Send, U: Send + Clone>(
        &self,
        v: &'a mut [T],
        init: U,
        op: impl Fn(&mut U, &'a mut T) + Send + Sync,
    ) {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.install(|| par_for_each_with(v, init, op)),
            JxlThreadPoolImpl::None => {
                let mut init = init;
                v.iter_mut().for_each(|item| op(&mut init, item))
            }
        }
    }
}

#[cfg(feature = "rayon")]
fn par_for_each<T: Send>(
    it: impl rayon::iter::IntoParallelIterator<Item = T>,
    op: impl Fn(T) + Send + Sync,
) {
    use rayon::prelude::*;
    it.into_par_iter().for_each(op);
}

#[cfg(feature = "rayon")]
fn par_for_each_with<T: Send, U: Send + Clone>(
    it: impl rayon::iter::IntoParallelIterator<Item = T>,
    init: U,
    op: impl Fn(&mut U, T) + Send + Sync,
) {
    use rayon::prelude::*;
    it.into_par_iter().for_each_with(init, op);
}

impl<'scope> JxlScope<'_, 'scope> {
    /// Spanws the given closure in current fork-join scope.
    pub fn spawn(&self, op: impl for<'r> FnOnce(JxlScope<'r, 'scope>) + Send + 'scope) {
        match self.0 {
            #[cfg(feature = "rayon")]
            JxlScopeInner::Rayon(scope) => scope.spawn(|scope| {
                let scope = JxlScope(JxlScopeInner::Rayon(scope));
                op(scope)
            }),
            JxlScopeInner::None(_) => op(JxlScope(JxlScopeInner::None(Default::default()))),
        }
    }
}
