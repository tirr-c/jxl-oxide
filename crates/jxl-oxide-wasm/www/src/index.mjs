import './styles.css';
import sunsetLogoUrl from './assets/sunset_logo.jxl';

class WorkerPool {
  #workerPool = [];
  #queue = [];
  #remaining;

  constructor(maxConcurrentWorkers = 8) {
    this.#remaining = maxConcurrentWorkers;
  }

  async getWorker() {
    if (this.#remaining <= 0) {
      return new Promise(resolve => {
        this.#queue.push(resolve);
      });
    }

    this.#remaining -= 1;
    let worker = this.#workerPool.shift();
    if (!worker) {
      worker = new Worker('jxl-decode-worker.js');
    }
    return worker;
  }

  putWorker(worker) {
    worker.postMessage({ type: 'reset' });

    const maybeResolve = this.#queue.shift();
    if (maybeResolve) {
      maybeResolve(worker);
      return;
    }

    this.#workerPool.push(worker);
    this.#remaining += 1;
  }
}

const workerPool = new WorkerPool(8);

const workers = new Map();
async function registerWorker() {
  if ('serviceWorker' in navigator) {
    const registerPromise = navigator.serviceWorker
      .register('service-worker.js', { updateViaCache: 'imports' })
      .then(
        registration => {
          if (registration.active) {
            registration.addEventListener('updatefound', () => {
              const sw = registration.installing;
              sw.addEventListener('statechange', () => {
                const state = sw.state;
                if (state === 'installed') {
                  console.info('Service Worker update is available.');
                }
              });
            });
          }
        },
        err => {
          console.error(`Registration failed with ${err}`);
          throw err;
        },
      );

    if (!navigator.serviceWorker.controller) {
      await Promise.all([
        registerPromise,
        new Promise(resolve => {
          function handle() {
            resolve();
            navigator.serviceWorker.removeEventListener('controllerchange', handle);
          }

          navigator.serviceWorker.addEventListener('controllerchange', handle);
        }),
      ]);
    }

    navigator.serviceWorker.addEventListener('message', async ev => {
      const sw = ev.source;

      const data = ev.data;
      const id = data.id;
      if (id == null) {
        return;
      }

      if (!workers.has(id)) {
        const worker = await workerPool.getWorker();
        worker.addEventListener('message', ev => {
          const data = ev.data;
          switch (data.type) {
            case 'feed':
              sw.postMessage({ id, type: 'feed' });
              break;
            case 'image':
              sw.postMessage(
                { id, type: 'image', image: data.image },
                [data.image.buffer]
              );
              break;
            case 'error':
              sw.postMessage({
                id,
                type: 'error',
                message: data.message,
              });
              break;
          }
          return worker;
        });
        workers.set(id, worker);
      }

      const worker = workers.get(id);
      switch (data.type) {
        case 'done':
          workers.delete(id);
          workerPool.putWorker(worker);
          break;
        case 'feed':
          worker.postMessage({ type: 'feed', buffer: data.buffer }, [data.buffer.buffer]);
          break;
        case 'decode':
          worker.postMessage({ type: 'decode' });
          break;
      }
    });
  }
}

async function decodeIntoImageNode(file, imgNode) {
  imgNode.classList.add('loading');

  const worker = await workerPool.getWorker();

  try {
    const blob = await new Promise((resolve, reject) => {
      worker.addEventListener('message', ev => {
        const data = ev.data;
        switch (data.type) {
          case 'blob':
            resolve(data.blob);
            break;
          case 'error':
            reject(new Error(data.message));
            break;
        }
      });
      worker.postMessage({ type: 'file', file });
    });

    const prevUrl = imgNode.src;
    if (prevUrl.startsWith('blob:')) {
      URL.revokeObjectURL(prevUrl);
    }
    imgNode.src = URL.createObjectURL(blob);
  } finally {
    imgNode.classList.remove('loading');
    workerPool.putWorker(worker);
  }
}

registerWorker().then(async () => {
  const container = document.getElementById('container');
  const form = container.querySelector('.form');
  const fileInput = container.querySelector('.file');

  const img = document.createElement('img');
  img.className = 'image';
  img.src = sunsetLogoUrl;
  await img.decode().catch(() => {});

  container.appendChild(img);
  form.addEventListener('submit', ev => {
    ev.preventDefault();

    const file = fileInput.files[0];
    if (file) {
      decodeIntoImageNode(file, img);
    }
  });
});
