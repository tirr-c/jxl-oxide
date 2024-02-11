const jxlOxidePromise = import('./wasm/jxl-oxide').then(jxlOxide => {
  jxlOxide.init();
  return jxlOxide;
});

async function handleRequest(request) {
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) {
    // Cross-origin request, forward
    return fetch(request);
  }

  const extensionIsJxl = url.pathname.endsWith('.jxl');
  const resp = await fetch(request);
  const status = resp.status;
  const respContentType = resp.headers.get('content-type')?.toLowerCase();
  const unknownContentType = !respContentType || respContentType === 'application/octet-stream';
  const isJxl = respContentType === 'image/jxl' || (extensionIsJxl && unknownContentType);
  if (!resp.ok || !isJxl) {
    return resp;
  }

  const { JxlImage } = await jxlOxidePromise;

  console.info(`Loading ${url.pathname}...`);
  const image = new JxlImage();

  const reader = resp.body.getReader();
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) {
      break;
    }
    image.feedBytes(chunk.value);
  }

  const loadingDone = image.tryInit();
  if (!loadingDone) {
    console.error('Partial image, no frame data');
    image.free();
    return new Response(new Uint8Array(), {
      status,
      headers: {
        'content-type': 'application/octet-stream',
      },
    });
  }

  console.info('Rendering...');
  const renderResult = image.render();
  image.free();

  console.info('Converting to PNG...');
  const output = renderResult.encodeToPng();

  console.info('Done! Transferring...');
  const headers = new Headers(resp.headers);
  headers.delete('content-length');
  headers.delete('content-encoding');
  headers.delete('transfer-encoding');
  headers.append('content-type', 'image/png');
  return new Response(output, {
    status,
    headers,
  });
}

self.addEventListener('install', ev => {
  self.skipWaiting();
  ev.waitUntil(jxlOxidePromise.then(() => {}));
});

self.addEventListener('active', ev => {
  ev.waitUntil(clients.claim());
});

self.addEventListener('fetch', ev => {
  ev.respondWith(handleRequest(ev.request));
});
