import { WasmSender } from './wasm-coordinator.js';
import { encodeText32Interleaved, MAX_MESSAGE_BYTES_32 } from './ook-text125.js';

const $ = (selector) => document.querySelector(selector);
const form = $('#tx-form');
const input = $('#message');
const startButton = $('#transmit-button');
const stopButton = $('#stop-button');
const count = $('#packet-count');
const delay = $('#worker-delay');
const byteCount = $('#byte-count');
const status = $('#tx-status');
const state = $('#engine-state');
const progress = $('#progress-bar');
const encoder = new TextEncoder();
let sender = null;
let active = false;
let packets = 0;
let runToken = 0;

function setStatus(message) { status.textContent = message; }
function refreshBytes() {
  const length = encoder.encode(input.value).length;
  byteCount.textContent = `${length} / ${MAX_MESSAGE_BYTES_32} bytes`;
  byteCount.style.color = length > MAX_MESSAGE_BYTES_32 ? '#ff9a91' : '';
  startButton.disabled = active || length < 1 || length > MAX_MESSAGE_BYTES_32;
}
function stop(reason = 'Stopped.') {
  if (!active) return;
  active = false;
  runToken++;
  sender?.stop();
  state.textContent = 'Workers stopping';
  stopButton.disabled = true;
  progress.style.width = '0';
  setStatus(reason);
  refreshBytes();
}
async function transmit(event) {
  event.preventDefault();
  if (active) return;
  const message = input.value;
  let packet;
  try { packet = encodeText32Interleaved(message); }
  catch (error) { setStatus(error.message); return; }
  active = true;
  const token = ++runToken;
  startButton.disabled = true;
  stopButton.disabled = false;
  input.disabled = true;
  packets = 0;
  count.textContent = '0';
  try {
    if (!sender) {
      state.textContent = 'Loading WASM';
      setStatus('Preparing four memory workers…');
      sender = new WasmSender({ workers: 4, memoryMiB: 16 });
      await sender.initialize();
    }
    state.textContent = 'Workers ready';
    while (active && token === runToken) {
      setStatus(`Transmitting packet ${packets + 1}…`);
      progress.style.width = '14%';
      const result = await sender.transmit(packet.chips, packet.chipMs,
        { leadMs: packet.leadMs, label: 'repeated-message' });
      if (!active || token !== runToken || result.type !== 'complete') break;
      packets++;
      count.textContent = String(packets);
      delay.textContent = `${result.maxChipLateMs.toFixed(1)} ms`;
      progress.style.width = '100%';
      await new Promise(resolve => setTimeout(resolve, 250));
      progress.style.width = '0';
    }
    if (token === runToken) setStatus(`Stopped after ${packets} complete packet${packets === 1 ? '' : 's'}.`);
  } catch (error) {
    setStatus(`Transmission stopped: ${error.message}`);
    sender?.terminate();
    sender = null;
  } finally {
    if (token === runToken) {
      active = false;
      input.disabled = false;
      stopButton.disabled = true;
      state.textContent = sender ? 'Workers ready' : 'Workers idle';
      refreshBytes();
    } else {
      input.disabled = false;
      state.textContent = sender ? 'Workers ready' : 'Workers idle';
      refreshBytes();
    }
  }
}
input.addEventListener('input', refreshBytes);
form.addEventListener('submit', transmit);
stopButton.addEventListener('click', () => stop(`Stopped after ${packets} complete packet${packets === 1 ? '' : 's'}.`));
document.addEventListener('visibilitychange', () => { if (document.hidden) stop('Stopped because the page is hidden.'); });
window.addEventListener('pagehide', () => { stop('Stopped because the page closed.'); sender?.terminate(); sender = null; });
refreshBytes();

if ('serviceWorker' in navigator) navigator.serviceWorker.register('./sw.js').catch(() => {});
