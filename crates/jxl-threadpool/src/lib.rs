#[derive(Debug, Clone)]
pub struct JxlThreadPool(JxlThreadPoolImpl);

#[derive(Debug, Clone)]
enum JxlThreadPoolImpl {
    #[cfg(feature = "rayon")]
    Rayon(std::sync::Arc<rayon_core::ThreadPool>),
    None,
}

#[derive(Debug, Copy, Clone)]
pub struct JxlScope<'r, 'scope>(JxlScopeInner<'r, 'scope>);

#[derive(Debug, Copy, Clone)]
enum JxlScopeInner<'r, 'scope> {
    #[cfg(feature = "rayon")]
    Rayon(&'r rayon_core::Scope<'scope>),
    None(std::marker::PhantomData<&'r &'scope ()>),
}

#[cfg(feature = "rayon")]
impl Default for JxlThreadPool {
    fn default() -> Self {
        let num_threads = std::thread::available_parallelism();
        let num_threads = match num_threads {
            Ok(num_threads) => num_threads.into(),
            Err(e) => {
                tracing::warn!(%e, "Failed to query available parallelism; falling back to single-threaded");
                return Self::none();
            },
        };

        let inner = rayon_core::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map(|pool| JxlThreadPoolImpl::Rayon(std::sync::Arc::new(pool)));

        match inner {
            Ok(inner) => {
                tracing::debug!(num_threads, "Initialized Rayon thread pool");
                Self(inner)
            },
            Err(e) => {
                tracing::warn!(%e, "Failed to initialize thread pool; falling back to single-threaded");
                Self::none()
            },
        }
    }
}

#[cfg(not(feature = "rayon"))]
impl Default for JxlThreadPool {
    fn default() -> Self {
        tracing::debug!("Not built with multithread support");
        Self::none()
    }
}

impl JxlThreadPool {
    pub const fn none() -> Self {
        Self(JxlThreadPoolImpl::None)
    }

    #[cfg(feature = "rayon")]
    pub fn rayon(pool: std::sync::Arc<rayon_core::ThreadPool>) -> Self {
        Self(JxlThreadPoolImpl::Rayon(pool))
    }

    pub fn is_multithreaded(&self) -> bool {
        match self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(_) => true,
            JxlThreadPoolImpl::None => false,
        }
    }
}

impl JxlThreadPool {
    pub fn spawn(&self, op: impl FnOnce() + Send + 'static) {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => pool.spawn(op),
            JxlThreadPoolImpl::None => op(),
        }
    }

    pub fn scope<'scope, R: Send>(
        &'scope self,
        op: impl for<'r> FnOnce(JxlScope<'r, 'scope>) -> R + Send,
    ) -> R {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(pool) => {
                pool.scope(|scope| {
                    let scope = JxlScope(JxlScopeInner::Rayon(scope));
                    op(scope)
                })
            },
            JxlThreadPoolImpl::None => {
                op(JxlScope(JxlScopeInner::None(Default::default())))
            },
        }
    }

    pub fn yield_now(&self) -> Option<JxlYield> {
        match &self.0 {
            #[cfg(feature = "rayon")]
            JxlThreadPoolImpl::Rayon(_) => rayon_core::yield_now().map(From::from),
            JxlThreadPoolImpl::None => None,
        }
    }
}

impl<'scope> JxlScope<'_, 'scope> {
    pub fn spawn(&self, op: impl for<'r> FnOnce(JxlScope<'r, 'scope>) + Send + 'scope) {
        match self.0 {
            #[cfg(feature = "rayon")]
            JxlScopeInner::Rayon(scope) => {
                scope.spawn(|scope| {
                    let scope = JxlScope(JxlScopeInner::Rayon(scope));
                    op(scope)
                })
            },
            JxlScopeInner::None(_) => {
                op(JxlScope(JxlScopeInner::None(Default::default())))
            },
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum JxlYield {
    Executed,
    Idle,
}

#[cfg(feature = "rayon")]
impl From<rayon_core::Yield> for JxlYield {
    fn from(value: rayon_core::Yield) -> Self {
        match value {
            rayon_core::Yield::Executed => Self::Executed,
            rayon_core::Yield::Idle => Self::Idle,
        }
    }
}
