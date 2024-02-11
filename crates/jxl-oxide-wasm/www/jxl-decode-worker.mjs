const jxlOxidePromise = import('./wasm/jxl-oxide.js').then(jxlOxide => {
  jxlOxide.init();
  return jxlOxide;
});

let image;
async function feed(buffer) {
  if (!image) {
    const { JxlImage } = await jxlOxidePromise;
    image = new JxlImage();
  }
  image.feedBytes(buffer);
}

function render() {
  const loadingDone = image.tryInit();
  if (!loadingDone) {
    image.free();
    throw new Error('Partial image, no frame data');
  }

  console.info('Rendering...');
  const renderResult = image.render();
  image.free();

  console.info('Converting to PNG...');
  const output = renderResult.encodeToPng();
  return output;
}

async function decodeFile(file) {
  const { JxlImage } = await jxlOxidePromise;
  image = new JxlImage();

  const reader = file.stream().getReader();
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) {
      break;
    }

    image.feedBytes(chunk.value);
  }

  const buffer = render();
  const blob = new File(
    [buffer],
    file.name + '.rendered.png',
    { type: 'image/png' },
  );
  self.postMessage({ type: 'blob', blob });
}

async function handleMessage(ev) {
  const data = ev.data;
  try {
    switch (data.type) {
      case 'file':
        await decodeFile(data.file);
        break;
      case 'feed':
        await feed(data.buffer);
        self.postMessage({ type: 'feed' });
        break;
      case 'decode': {
        const image = render();
        self.postMessage(
          { type: 'image', image },
          [image.buffer],
        );
        break;
      }
    }
  } catch (err) {
    self.postMessage({
      type: 'error',
      message: String(err),
    });
  }
}

self.addEventListener('message', ev => {
  handleMessage(ev);
})
