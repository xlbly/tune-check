// Runs model download, separation and pitch tracking off the main thread.
// Memory matters most here: iOS Safari kills a tab that goes much past ~1 GB, so only one
// model is loaded at a time and ONNX Runtime's memory arena (which keeps peak-sized blocks
// around) is turned off. Measured in Node: 1.5 GB -> 0.2 GB peak for the larger model.
import * as ort from "./vendor/ort.webgpu.min.mjs";
import { analyze } from "./lib/pipeline.js";
import { MODELS, setSegmentFrames } from "./lib/mdx.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = "tune-check-models-v2";   // v2: models accept any segment length
caches.delete("tune-check-models-v1").catch(() => {});
const FILES = [MODELS.vocals.file, MODELS.karaoke.file];
const TOTAL_BYTES = 119.5e6;
let backend = null;          // execution provider of the latest session
let wasmStarted = false;     // thread count can only be set before the runtime starts

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const modelUrl = (path) => new URL(`./models/${path}`, import.meta.url).href;

async function openCache() { try { return await caches.open(CACHE); } catch { return null; } }

// Make sure a model is in the Cache API, downloading it with progress if needed.
async function download(path, onBytes) {
  const cache = await openCache(), url = modelUrl(path);
  if (cache && (await cache.match(url))) return;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${path} (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onBytes(got, total);
  }
  if (cache) { try { await cache.put(url, new Response(new Blob(parts), { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota: fetched again next time */ } }
}

async function modelBytes(path) {
  const cache = await openCache(), url = modelUrl(path);
  const hit = cache && (await cache.match(url));
  const res = hit || (await fetch(url));
  return new Uint8Array(await res.arrayBuffer());
}

const SESSION_OPTS = { graphOptimizationLevel: "all", enableCpuMemArena: false, enableMemPattern: false, executionMode: "sequential" };

// Fast mode: half-precision models on the GPU, only if the GPU supports 16-bit floats.
let gpuF16 = null;
async function gpuHasF16() {
  if (gpuF16 === null) {
    try { const a = await self.navigator.gpu?.requestAdapter(); gpuF16 = !!a && a.features.has("shader-f16"); } catch { gpuF16 = false; }
  }
  return gpuF16;
}

async function createSession(cfg, fast) {
  if (fast && (await gpuHasF16())) {
    try {
      const path = `fp16/${cfg.file}`;
      await download(path, (g, t) => post({ type: "progress", label: "Downloading the fast-mode models", frac: t ? g / t : 0 }));
      const s = await ort.InferenceSession.create(await modelBytes(path), { ...SESSION_OPTS, executionProviders: ["webgpu"] });
      return { s, backend: "webgpu" };
    } catch (e) { post({ type: "note", message: `Fast mode is unavailable here (${e.message || e}); using the CPU.` }); }
  } else if (fast) {
    post({ type: "note", message: "Fast mode needs a GPU with 16-bit support, which this browser doesn't offer; using the CPU." });
  }
  const s = await ort.InferenceSession.create(await modelBytes(cfg.file), { ...SESSION_OPTS, executionProviders: ["wasm"] });
  return { s, backend: "wasm" };
}

async function load() {
  const got = {};
  for (const f of FILES) {
    await download(f, (g) => { got[f] = g; post({ type: "download", got: Object.values(got).reduce((a, b) => a + b, 0), total: TOTAL_BYTES }); });
  }
  post({ type: "ready" });
}

async function runnerFor(cfg, fast) {
  const made = await createSession(cfg, fast);
  let s = made.s;
  backend = made.backend;
  post({ type: "backend", backend });
  const run = async (data, dims) => {
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", data, dims) });
    const t = out[s.outputNames[0]];
    const d = t.getData ? await t.getData() : t.data;
    t.dispose?.();
    return d;
  };
  run.release = async () => { if (s) { await s.release(); s = null; } };
  return run;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") await load();
    if (m.type === "analyze") {
      setSegmentFrames(m.segmentFrames || 128);
      if (!wasmStarted && self.crossOriginIsolated) ort.env.wasm.numThreads = Math.min(m.threads || 4, navigator.hardwareConcurrency || 1);
      wasmStarted = true;
      const t0 = performance.now();
      const r = await analyze(m.stereo, (cfg) => runnerFor(cfg, m.fast), (label, frac) => post({ type: "progress", label, frac }));
      post({ type: "result", seconds: (performance.now() - t0) / 1000, backend, r },
        [r.fUser.buffer, r.fSinger.buffer, r.dev.buffer, r.smooth.buffer, r.userAudio.buffer, r.singerAudio.buffer]);
    }
  } catch (err) {
    post({ type: "error", message: String(err && err.message || err) });
  }
};
