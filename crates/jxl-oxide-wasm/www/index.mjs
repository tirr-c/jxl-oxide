import './styles.css';

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
    try {
      await navigator.serviceWorker.register('/service-worker.js', { scope: '/' });
    } catch (error) {
      console.error(`Registration failed with ${error}`);
    }
  }

  navigator.serviceWorker.addEventListener('message', async ev => {
    const sw = ev.source;

    const data = ev.data;
    const id = data.id;
    if (!id) {
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

const template = `
<form class="form">
<label>
Attach a file: <input type="file" class="file" accept=".jxl,image/jxl">
</label>
<input type="submit" value="Load">
</form>
`;

registerWorker().then(async () => {
  const container = document.createElement('main');
  container.id = 'container';
  container.innerHTML = template;
  document.body.appendChild(container);

  const form = container.querySelector('.form');
  const fileInput = container.querySelector('.file');

  const img = document.createElement('img');
  img.className = 'image';
  img.src = '/assets/sunset_logo.jxl';
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
