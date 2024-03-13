import init, { runThread } from '../../../';

self.addEventListener('message', async ev => {
  const { module, memory, receiver } = ev.data;
  await init(module, memory);
  runThread(receiver);
});
