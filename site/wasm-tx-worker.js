/* WebAssembly memory sender prototype. Preload online; run without network. */
"use strict";

const MAX_RUN_MS = 15_000;
const BUFFER_BASE = 65_536;
const CALL_STEPS = 4_096;
let wasm = null;
let bufferBytes = 0;
let workerIndex = -1;
let running = false;
let stopRequested = false;

function clockMs() {
  const origin = Number(performance.timeOrigin);
  return Number.isFinite(origin) ? origin + performance.now() : Date.now();
}

function pause(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function waitUntil(deadline) {
  while (!stopRequested) {
    const remaining = deadline - clockMs();
    if (remaining <= 0) return;
    await pause(Math.min(10, Math.max(1, remaining)));
  }
}

async function initialize(request) {
  if (wasm || running) throw new Error("Worker already initialized");
  const memoryMiB = Number(request.memoryMiB);
  if (![8, 16, 32].includes(memoryMiB)) throw new RangeError("Memory size must be 8, 16, or 32 MiB");
  workerIndex = Number(request.workerIndex);
  if (!Number.isInteger(workerIndex) || workerIndex < 0 || workerIndex > 3) {
    throw new RangeError("Worker index must be 0–3");
  }
  const wasmUrl = request.wasmUrl || "./memstress.wasm";
  const response = await fetch(wasmUrl, { cache: "force-cache" });
  if (!response.ok) throw new Error(`WASM load failed: HTTP ${response.status}`);
  const source = await response.arrayBuffer();
  const instance = (await WebAssembly.instantiate(source)).instance;
  const exports = instance.exports;
  if (!exports.memory || !exports.touch || !exports.get_checksum) throw new Error("WASM exports are incomplete");
  bufferBytes = memoryMiB * 1024 * 1024;
  const requiredPages = Math.ceil((BUFFER_BASE + bufferBytes) / 65_536);
  const existingPages = exports.memory.buffer.byteLength / 65_536;
  if (requiredPages > existingPages) exports.memory.grow(requiredPages - existingPages);
  new Uint8Array(exports.memory.buffer, BUFFER_BASE, bufferBytes).fill(0);
  wasm = exports;
  postMessage({ type: "ready", workerIndex, memoryMiB, wasmBytes: source.byteLength });
}

async function run(request) {
  if (!wasm) throw new Error("WASM module is not ready");
  if (running) throw new Error("Worker is already running");
  const chips = String(request.chips ?? "");
  const chipMs = Number(request.chipMs);
  const label = String(request.label ?? "pattern");
  if (!/^[01]+$/.test(chips) || chips.length > 512) throw new RangeError("chips must be 1–512 binary symbols");
  if (!Number.isFinite(chipMs) || chipMs < 20 || chipMs > 1_000) {
    throw new RangeError("chipMs must be 20–1000");
  }
  const startAtMs = Number(request.startAtEpochMs);
  const initialLeadMs = startAtMs - clockMs();
  const plannedMs = initialLeadMs + chips.length * chipMs;
  if (!Number.isFinite(startAtMs) || initialLeadMs < 100 || initialLeadMs > 3_000 || plannedMs > MAX_RUN_MS) {
    throw new RangeError("Invalid or stale synchronized start time");
  }
  running = true;
  stopRequested = false;
  try {
    wasm.reset_checksum(workerIndex + 1);
    let cursor = 0;
    let updates = 0;
    let activeMs = 0;
    let maxLateMs = 0;
    const begunAt = clockMs();
    postMessage({ type: "started", workerIndex, label, startAtMs, chipMs,
                  chips: chips.length, plannedMs });
    await waitUntil(startAtMs);
    for (let chip = 0; chip < chips.length && !stopRequested; chip++) {
      const target = startAtMs + chip * chipMs;
      await waitUntil(target);
      if (stopRequested) break;
      const actual = clockMs();
      maxLateMs = Math.max(maxLateMs, actual - target);
      const deadline = target + chipMs;
      if (chips[chip] === "1") {
        const activeStart = clockMs();
        while (clockMs() < deadline - 5 && !stopRequested) {
          cursor = wasm.touch(BUFFER_BASE, bufferBytes, cursor, CALL_STEPS);
          updates += CALL_STEPS;
        }
        activeMs += clockMs() - activeStart;
      }
      await waitUntil(deadline);
      if (clockMs() - begunAt > MAX_RUN_MS) throw new Error("15-second safety limit reached");
    }
    postMessage({ type: stopRequested ? "stopped" : "complete", workerIndex,
                  label, elapsedMs: clockMs() - begunAt, plannedMs, chipMs,
                  chips: chips.length, memoryMiB: bufferBytes / 1_048_576,
                  arrayUpdates: updates, activeMs, maxLateMs,
                  checksum: wasm.get_checksum() >>> 0 });
  } finally {
    running = false;
    stopRequested = false;
  }
}

self.onmessage = event => {
  const request = event.data || {};
  if (request.type === "stop") {
    stopRequested = true;
  } else if (request.type === "init") {
    initialize(request).catch(error => postMessage({ type: "error", workerIndex,
      message: error instanceof Error ? error.message : String(error) }));
  } else if (request.type === "run") {
    if (running) {
      postMessage({ type: "error", workerIndex, message: "Worker is already running" });
      return;
    }
    run(request).catch(error => postMessage({ type: "error", workerIndex,
      message: error instanceof Error ? error.message : String(error) }));
  }
};
