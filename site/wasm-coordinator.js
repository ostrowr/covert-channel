/* Coordinate 1/2/4 independent WASM workers without SharedArrayBuffer. */
"use strict";

export class WasmSender {
  constructor({ workers = 2, memoryMiB = 8,
                workerUrl = "./wasm-tx-worker.js", wasmUrl = "./memstress.wasm" } = {}) {
    if (![1, 2, 4].includes(workers)) throw new RangeError("workers must be 1, 2, or 4");
    if (![8, 16, 32].includes(memoryMiB)) throw new RangeError("memoryMiB must be 8, 16, or 32");
    if (workers * memoryMiB > 128) throw new RangeError("total WASM memory exceeds 128 MiB");
    this.count = workers;
    this.memoryMiB = memoryMiB;
    this.workerUrl = workerUrl;
    this.wasmUrl = wasmUrl;
    this.workers = [];
    this.ready = false;
    this.running = false;
  }

  async initialize() {
    if (this.workers.length) throw new Error("Sender already initialized");
    try {
      const ready = [];
      for (let index = 0; index < this.count; index++) {
        const worker = new Worker(this.workerUrl);
        this.workers.push(worker);
        ready.push(new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error(`Worker ${index} did not load WASM`)), 10_000);
          worker.addEventListener("message", function onReady(event) {
            if (event.data?.workerIndex !== index) return;
            if (event.data.type === "ready" || event.data.type === "error") {
              clearTimeout(timeout);
              worker.removeEventListener("message", onReady);
              event.data.type === "ready" ? resolve(event.data) : reject(new Error(event.data.message));
            }
          });
          worker.addEventListener("error", error => reject(error));
        }));
        worker.postMessage({ type: "init", workerIndex: index, memoryMiB: this.memoryMiB,
                             wasmUrl: this.wasmUrl });
      }
      const details = await Promise.all(ready);
      this.ready = true;
      return details;
    } catch (error) {
      this.terminate();
      throw error;
    }
  }

  async transmit(chips, chipMs, { leadMs = 750, label = "pattern" } = {}) {
    if (!this.ready) throw new Error("Workers must load before transmission");
    if (this.running) throw new Error("A transmission is already running");
    if (typeof chips !== "string" || !/^[01]+$/.test(chips) || chips.length > 512) {
      throw new RangeError("chips must be a 1–512-character binary string");
    }
    if (!Number.isFinite(chipMs) || chipMs < 20 || chipMs > 1_000) {
      throw new RangeError("chipMs must be 20–1000");
    }
    if (!Number.isFinite(leadMs) || leadMs < 500 || leadMs > 2_000) {
      throw new RangeError("leadMs must be between 500 and 2000");
    }
    if (leadMs + chips.length * chipMs > 15_000) {
      throw new RangeError("pattern exceeds the 15-second limit");
    }
    this.running = true;
    const startAtEpochMs = performance.timeOrigin + performance.now() + leadMs;
    const completions = this.workers.map((worker, index) => new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error(`Worker ${index} timed out`)), 20_000);
      const onMessage = event => {
        if (event.data?.workerIndex !== index) return;
        if (!["complete", "stopped", "error"].includes(event.data.type)) return;
        clearTimeout(timeout);
        worker.removeEventListener("message", onMessage);
        event.data.type === "error" ? reject(new Error(event.data.message)) : resolve(event.data);
      };
      worker.addEventListener("message", onMessage);
    }));
    try {
      this.workers.forEach(worker => worker.postMessage({ type: "run", chips, chipMs,
                                                          label, startAtEpochMs }));
      const results = await Promise.all(completions);
      return {
        type: results.every(item => item.type === "complete") ? "complete" : "stopped",
        workers: this.count,
        memoryMiBPerWorker: this.memoryMiB,
        totalMemoryMiB: this.count * this.memoryMiB,
        startAtEpochMs,
        maxChipLateMs: Math.max(...results.map(item => item.maxLateMs)),
        totalArrayUpdates: results.reduce((sum, item) => sum + item.arrayUpdates, 0),
        results,
      };
    } catch (error) {
      this.stop();
      throw error;
    } finally {
      this.running = false;
    }
  }

  stop() {
    for (const worker of this.workers) worker.postMessage({ type: "stop" });
  }

  terminate() {
    this.stop();
    for (const worker of this.workers) worker.terminate();
    this.workers = [];
    this.ready = false;
    this.running = false;
  }
}
