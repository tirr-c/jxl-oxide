use jxl_oxide::JxlThreadPool;
use wasm_bindgen::prelude::*;
use web_sys::js_sys;

fn _emit_worker() {
    wasm_bindgen::link_to!(module = "/src/worker.js");
}

#[wasm_bindgen]
pub struct ThreadPool {
    pub(crate) inner: JxlThreadPool,
    workers: JsValue,
}

#[wasm_bindgen]
impl ThreadPool {
    #[wasm_bindgen(setter)]
    #[doc(hidden)]
    pub fn set_workers(&mut self, workers: JsValue) {
        self.workers = workers;
    }
}

#[wasm_bindgen(js_name = initThreadPool)]
pub fn init_thread_pool(num_threads: usize) -> ThreadPoolPromise {
    let mut rx_list = Vec::new();
    let mut promise_list = Vec::new();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .spawn_handler(|thread| {
            let (tx, rx) = std::sync::mpsc::sync_channel(0);
            rx_list.push(rx);
            let rx = rx_list.last().unwrap();
            let promise =
                post_message_to_spawner(wasm_bindgen::module(), wasm_bindgen::memory(), rx);
            promise_list.push(promise);
            tx.send(thread).unwrap();
            Ok(())
        })
        .build()
        .map_err(|e| e.to_string())
        .unwrap_throw();
    let pool = JxlThreadPool::with_rayon_thread_pool(std::sync::Arc::new(pool));
    return_after_wait(
        promise_list,
        ThreadPool {
            inner: pool,
            workers: JsValue::null(),
        },
    )
}

/// # Safety
/// `receiver` must be a receiver from message sent by `initThreadPool`.
#[wasm_bindgen(js_name = runThread)]
pub unsafe fn run_thread(receiver: *const std::sync::mpsc::Receiver<rayon::ThreadBuilder>) {
    let rx = &*receiver;
    let thread = rx.recv().unwrap_throw();
    thread.run()
}

#[wasm_bindgen(typescript_custom_section)]
const THREAD_POOL_PROMISE: &str = r"
type ThreadPoolPromise = Promise<ThreadPool>;
";

#[wasm_bindgen(module = "/src/helper.js")]
extern "C" {
    #[wasm_bindgen(typescript_type = "ThreadPoolPromise")]
    pub type ThreadPoolPromise;

    fn post_message_to_spawner(
        module: JsValue,
        memory: JsValue,
        receiver: *const std::sync::mpsc::Receiver<rayon::ThreadBuilder>,
    ) -> js_sys::Promise;

    fn return_after_wait(promises: Vec<js_sys::Promise>, pool: ThreadPool) -> ThreadPoolPromise;
}
