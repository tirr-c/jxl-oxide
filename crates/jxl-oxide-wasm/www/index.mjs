import './styles.css';

const workers = new Map();
async function registerWorker() {
  if ('serviceWorker' in navigator) {
    try {
      await navigator.serviceWorker.register('/service-worker.js', { scope: '/' });
    } catch (error) {
      console.error(`Registration failed with ${error}`);
    }
  }

  navigator.serviceWorker.addEventListener('message', ev => {
    const sw = ev.source;

    const data = ev.data;
    const id = data.id;
    if (!id) {
      return;
    }

    if (!workers.has(id)) {
      const worker = new Worker('jxl-decode-worker.js');
      worker.addEventListener('message', ev => {
        const data = ev.data;
        switch (data.type) {
          case 'feed':
            sw.postMessage({ id, type: 'feed' });
            break;
          case 'image':
            sw.postMessage(
              { id, type: 'image', image: data.image },
              [data.image]
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
      });
      workers.set(id, worker);
    }

    const worker = workers.get(id);
    switch (data.type) {
      case 'done':
        worker.terminate();
        workers.delete(id);
        break;
      case 'feed':
        worker.postMessage({ type: 'feed', buffer: data.buffer }, [data.buffer]);
        break;
      case 'decode':
        worker.postMessage({ type: 'decode' });
        break;
    }
  });
}

registerWorker().then(() => {
  const container = document.createElement('main');
  container.id = 'container';

  const input = document.createElement('input');
  input.value = '/assets/sunset_logo.jxl';
  const img = new Image();
  img.className = 'image';
  img.src = '/assets/sunset_logo.jxl';

  const form = document.createElement('form');
  form.className = 'form';
  const label = document.createElement('label');
  label.appendChild(document.createTextNode('JPEG XL image path: '));
  label.appendChild(input);
  form.appendChild(label);
  const submit = document.createElement('input');
  submit.type = 'submit';
  submit.value = 'Load';
  form.appendChild(submit);
  form.addEventListener('submit', ev => {
    ev.preventDefault();
    const path = input.value;
    img.src = path;
  });

  container.appendChild(form);
  container.appendChild(img);
  document.body.appendChild(container);
});
