import './styles.css';

async function registerWorker() {
  if ('serviceWorker' in navigator) {
    try {
      await navigator.serviceWorker.register('/service-worker.js', { scope: '/' });
    } catch (error) {
      console.error(`Registration failed with ${error}`);
    }
  }
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
