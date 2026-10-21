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
let backend = null;          // execution provider that worked, reused for the next session

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const modelUrl = (file) => new URL(`./models/${file}`, import.meta.url).href;

async function openCache() { try { return await caches.open(CACHE); } catch { return null; } }

// Make sure a model is in the Cache API, downloading it with progress if needed.
async function download(file, onBytes) {
  const cache = await openCache(), url = modelUrl(file);
  if (cache && (await cache.match(url))) return;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${file} (HTTP ${res.status})`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); got += value.length; onBytes(got, total);
  }
  if (cache) { try { await cache.put(url, new Response(new Blob(parts), { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota: fetched again at analysis time */ } }
}

async function modelBytes(file) {
  const cache = await openCache(), url = modelUrl(file);
  const hit = cache && (await cache.match(url));
  const res = hit || (await fetch(url));
  return new Uint8Array(await res.arrayBuffer());
}

async function createSession(file, preferWasm) {
  const tries = backend ? [backend] : [...(self.navigator.gpu && !preferWasm ? ["webgpu"] : []), "wasm"];
  let bytes = await modelBytes(file), lastErr;
  for (const ep of tries) {
    try {
      const s = await ort.InferenceSession.create(bytes, {
        executionProviders: [ep], graphOptimizationLevel: "all",
        enableCpuMemArena: false, enableMemPattern: false, executionMode: "sequential",
      });
      backend = ep;
      bytes = null;
      return s;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function load() {
  const got = {};
  for (const f of FILES) {
    await download(f, (g) => { got[f] = g; post({ type: "download", got: Object.values(got).reduce((a, b) => a + b, 0), total: TOTAL_BYTES }); });
  }
  post({ type: "ready" });
}

async function runnerFor(cfg, preferWasm) {
  let s = await createSession(cfg.file, preferWasm);
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
      const t0 = performance.now();
      const r = await analyze(m.stereo, (cfg) => runnerFor(cfg, m.preferWasm), (label, frac) => post({ type: "progress", label, frac }));
      post({ type: "result", seconds: (performance.now() - t0) / 1000, backend, r },
        [r.fUser.buffer, r.fSinger.buffer, r.dev.buffer, r.smooth.buffer, r.userAudio.buffer, r.singerAudio.buffer]);
    }
  } catch (err) {
    post({ type: "error", message: String(err && err.message || err) });
  }
};
