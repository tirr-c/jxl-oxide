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

async function handleMessage(ev) {
  const {type, buffer} = ev.data;
  try {
    switch (type) {
      case 'feed':
        await feed(new Uint8Array(buffer));
        self.postMessage({ type: 'feed' });
        break;
      case 'decode': {
        const image = render();
        const buffer = image.buffer;
        self.postMessage(
          { type: 'image', image: buffer },
          [buffer],
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
