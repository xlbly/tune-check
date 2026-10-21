// Runs model download, separation and pitch tracking off the main thread.
import * as ort from "./vendor/ort.webgpu.min.mjs";
import { analyze } from "./lib/pipeline.js";
import { MODELS } from "./lib/mdx.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE = "tune-check-models-v1";
const sessions = {};
let backend = null;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);

// Download a model with progress, keeping it in the Cache API for next time.
async function fetchModel(file, onBytes) {
  const url = new URL(`./models/${file}`, import.meta.url).href;
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache API: download every time */ }
  const hit = cache && (await cache.match(url));
  if (hit) { const b = await hit.arrayBuffer(); onBytes(b.byteLength, b.byteLength); return b; }
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
  const buf = new Uint8Array(got);
  let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(buf.slice().buffer, { headers: { "content-type": "application/octet-stream" } })); } catch { /* quota */ } }
  return buf.buffer;
}

async function createSession(bytes) {
  const tries = [];
  if (self.navigator.gpu) tries.push("webgpu");
  tries.push("wasm");
  let lastErr;
  for (const ep of backend ? [backend] : tries) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: "all" });
      backend = ep;
      return s;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function load() {
  const files = [MODELS.vocals.file, MODELS.karaoke.file];
  const totals = {}, gots = {};
  for (const f of files) {
    const bytes = await fetchModel(f, (got, total) => {
      gots[f] = got; totals[f] = total || got;
      const g = Object.values(gots).reduce((a, b) => a + b, 0);
      post({ type: "download", got: g, total: 119.5e6 });
    });
    sessions[f] = await createSession(bytes);
  }
  post({ type: "ready", backend });
}

function runnerFor(cfg) {
  const s = sessions[cfg.file];
  return async (data, dims) => {
    const out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", data, dims) });
    return out[s.outputNames[0]].getData ? await out[s.outputNames[0]].getData() : out[s.outputNames[0]].data;
  };
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "load") await load();
    if (m.type === "analyze") {
      const t0 = performance.now();
      const r = await analyze(m.stereo, async (cfg) => runnerFor(cfg), (label, frac) => post({ type: "progress", label, frac }));
      post({ type: "result", seconds: (performance.now() - t0) / 1000, backend, r },
        [r.fUser.buffer, r.fSinger.buffer, r.dev.buffer, r.smooth.buffer, r.userAudio.buffer, r.singerAudio.buffer]);
    }
  } catch (err) {
    post({ type: "error", message: String(err && err.message || err) });
  }
};
