let nextId = 0;
const resultPromiseMap = new Map();

async function handleRequest(ev) {
  const request = ev.request;
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

  console.info(`Loading ${url.pathname}...`);
  const client = await self.clients.get(ev.clientId);
  const id = nextId;
  nextId += 1;

  const reader = resp.body.getReader();
  while (true) {
    const chunk = await reader.read();
    if (chunk.done) {
      break;
    }

    const buffer = chunk.value.buffer;
    await new Promise((resolve, reject) => {
      resultPromiseMap.set(id, [resolve, reject]);
      client.postMessage({ id, type: 'feed', buffer }, [buffer]);
    });
  }

  const output = await new Promise((resolve, reject) => {
    resultPromiseMap.set(id, [resolve, reject]);
    client.postMessage({ id, type: 'decode' });
  });

  client.postMessage({ id, type: 'done' });

  console.info('Transferring...');
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

self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('active', ev => {
  ev.waitUntil(clients.claim());
});

self.addEventListener('fetch', ev => {
  ev.respondWith(handleRequest(ev));
});

self.addEventListener('message', ev => {
  const data = ev.data;
  const id = data.id;

  const cbs = resultPromiseMap.get(id);
  if (!cbs) {
    return;
  }
  const [resolve, reject] = cbs;
  resultPromiseMap.delete(id);

  switch (data.type) {
    case 'feed':
      resolve();
      break;
    case 'image':
      resolve(data.image);
      break;
    case 'error':
      reject(new Error(data.message));
      break;
  }
});
