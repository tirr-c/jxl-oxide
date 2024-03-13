export async function postMessageToSpawner(module, memory, receiver) {
  const worker = new Worker(new URL('./worker.js', import.meta));
  const message = { module, memory, receiver };
  return new Promise((resolve, reject) => {
    worker.addEventListener('message', () => {
      resolve(worker);
    }, { once: true });
    worker.addEventListener('error', ev => {
      reject(ev.error);
    }, { once: true });

    worker.postMessage(message);
  });
}

export async function returnAfterWait(promises, pool) {
  const workers = await Promise.all(promises);
  pool.workers = workers;
  return pool;
}
