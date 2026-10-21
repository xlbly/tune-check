// UI: record or load a snippet, hand it to the worker, draw the result, play it back.

const $ = (id) => document.getElementById(id);
const recBtn = $("recBtn"), timerEl = $("timer"), hint = $("hint");
const statusBox = $("status"), statusLabel = $("statusLabel"), statusBar = $("statusBar"), statusNote = $("statusNote");
const MAX_SECONDS = 60, SR = 44100;

let worker, ready = false, busy = false;
let audioCtx = null, mediaRec = null, stream = null, chunks = [], recStart = 0, tick = null;
let last = null;            // { r, recording (Float32Array @44.1k), duration }
let playing = null;         // { src, start, offset }
let currentBackend = null;

// If the browser kills the tab mid-analysis (usually out of memory), the page reloads with
// no error. A marker saved during the analysis lets the next load say what happened, and
// switch from the GPU to the lighter CPU path if the GPU was in use.
const RUN_KEY = "tc-running", LEVEL_KEY = "tc-light-level";
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage blocked */ } },
  del(k) { try { localStorage.removeItem(k); } catch { /* storage blocked */ } },
};
// Lighter modes after a crash: 0 = GPU if available, 128-frame segments;
// 1 = CPU, 128 frames; 2 = CPU, 64 frames (about 30% less memory, slightly less accurate).
let level = Math.min(2, Number(store.get(LEVEL_KEY)) || 0);
// Phones never use the GPU path: on an iPhone it crashed the whole device.
const IS_PHONE = /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || (/Macintosh/.test(navigator.userAgent) && navigator.maxTouchPoints > 1);
const modeFor = (l) => ({ preferWasm: IS_PHONE || l >= 1, segmentFrames: l >= 2 ? 64 : 128, threads: IS_PHONE ? 2 : 4 });
let crashNotice = null;
try {
  const prev = JSON.parse(store.get(RUN_KEY) || "null");
  if (prev) {
    store.del(RUN_KEY);
    const next = prev.backend === "webgpu" ? Math.max(level, 1) : 2;
    const changed = next > level;
    level = next;
    store.set(LEVEL_KEY, String(level));
    crashNotice = `The last analysis stopped during "${prev.stage}", most likely because the browser ran out of memory. ` +
      (changed ? "It will now run in a lighter mode that needs less memory but takes longer." : "Try a shorter snippet (10–20 seconds), and close other tabs.");
  }
} catch { store.del(RUN_KEY); }

function ctx() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

function setStatus(label, frac, note) {
  statusBox.hidden = false;
  statusLabel.textContent = label;
  statusLabel.classList.remove("err");
  statusBar.style.width = `${Math.round((frac || 0) * 100)}%`;
  if (note !== undefined) statusNote.textContent = note;
}
function setError(msg) {
  statusBox.hidden = false;
  statusLabel.textContent = msg;
  statusLabel.classList.add("err");
  statusBar.style.width = "0";
}
function setIdle() {
  busy = false;
  recBtn.disabled = !ready;
  recBtn.textContent = "Record";
  recBtn.classList.remove("recording");
}

// ---- worker -------------------------------------------------------------
function startWorker() {
  worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  worker.onmessage = (e) => {
    const m = e.data;
    if (m.type === "download") setStatus("Downloading the models (about 120 MB, once)", m.got / m.total);
    if (m.type === "backend") currentBackend = m.backend;
    if (m.type === "ready") {
      ready = true;
      statusBox.hidden = true;
      setIdle();
      if (crashNotice) setError(crashNotice);
    }
    if (m.type === "progress") {
      store.set(RUN_KEY, JSON.stringify({ stage: m.label, backend: currentBackend }));
      const steps = ["Removing the instruments", "Separating your voice from the singer's", "Tracking pitch"];
      const i = Math.max(0, steps.indexOf(m.label));
      setStatus(`${m.label}…`, (i + m.frac) / steps.length, "This runs on your device and can take a minute.");
    }
    if (m.type === "result") { store.del(RUN_KEY); statusBox.hidden = true; showResult(m); setIdle(); }
    if (m.type === "error") { store.del(RUN_KEY); setError(`Something went wrong: ${m.message}`); setIdle(); }
  };
  worker.onerror = (e) => { setError(`The analysis could not start: ${e.message || "unknown error"}`); };
  worker.postMessage({ type: "load" });
}

// ---- audio in -------------------------------------------------------------
async function toStereo44k(arrayBuffer) {
  const decoded = await ctx().decodeAudioData(arrayBuffer);
  const len = Math.ceil(decoded.duration * SR);
  const off = new OfflineAudioContext(1, len, SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const out = await off.startRendering();
  const mono = out.getChannelData(0).slice(0, Math.min(len, MAX_SECONDS * SR));
  return mono;
}

async function analyzeBuffer(arrayBuffer) {
  busy = true;
  recBtn.disabled = true;
  $("results").hidden = true;
  setStatus("Preparing the audio…", 0, "This runs on your device and can take a minute.");
  try {
    const mono = await toStereo44k(arrayBuffer);
    if (mono.length < SR * 2) { setError("That snippet is too short. Record at least a few seconds."); setIdle(); return; }
    last = { recording: mono, duration: mono.length / SR };
    store.set(RUN_KEY, JSON.stringify({ stage: "Preparing the audio", backend: null }));
    worker.postMessage({ type: "analyze", stereo: [mono.slice(), mono.slice()], ...modeFor(level) });
  } catch (err) {
    setError(`Could not read that audio: ${err.message || err}`);
    setIdle();
  }
}

async function startRecording() {
  ctx();
  stopPlayback();
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
  } catch {
    setError("Microphone access was refused. Allow it in your browser settings, then try again.");
    return;
  }
  chunks = [];
  mediaRec = new MediaRecorder(stream);
  mediaRec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRec.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    const blob = new Blob(chunks, { type: mediaRec.mimeType });
    analyzeBuffer(await blob.arrayBuffer());
  };
  mediaRec.start();
  recStart = performance.now();
  recBtn.textContent = "Stop";
  recBtn.classList.add("recording");
  $("results").hidden = true;
  statusBox.hidden = true;
  tick = setInterval(() => {
    const s = (performance.now() - recStart) / 1000;
    timerEl.textContent = fmtTime(s);
    if (s >= MAX_SECONDS) stopRecording();
  }, 200);
}

function stopRecording() {
  clearInterval(tick);
  if (mediaRec && mediaRec.state !== "inactive") mediaRec.stop();
  recBtn.disabled = true;
  recBtn.classList.remove("recording");
  recBtn.textContent = "…";
}

recBtn.addEventListener("click", () => {
  if (!ready || busy && !(mediaRec && mediaRec.state === "recording")) return;
  if (mediaRec && mediaRec.state === "recording") stopRecording();
  else startRecording();
});

$("fileIn").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f || !ready || busy) return;
  ctx();
  stopPlayback();
  timerEl.textContent = "";
  analyzeBuffer(await f.arrayBuffer());
});

// ---- results --------------------------------------------------------------
const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const pct = (v) => (v == null ? "–" : `${Math.round(v)}%`);
const NOTE = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"];
const midi = (f) => 69 + 12 * Math.log2(f / 440);

function showResult(m) {
  const r = m.r;
  last = { ...last, r };
  $("sIn").textContent = pct(r.inTunePct);
  $("sSlight").textContent = pct(r.slightlyOffPct);
  $("sOff").textContent = pct(r.offPct);
  const box = $("stretchBox");
  box.textContent = "";
  if (r.comparedSeconds < 1) {
    box.innerHTML = "<p class='hint'>Not enough moments where both you and the singer could be heard. Try singing closer to the phone, with the song a bit louder.</p>";
  } else if (r.stretches.length) {
    const h = document.createElement("h2"); h.style.marginTop = "16px"; h.textContent = "Where you were off";
    const ul = document.createElement("ul"); ul.className = "stretches";
    for (const s of r.stretches) {
      const li = document.createElement("li");
      const semis = Math.abs(s.cents) / 100;
      li.textContent = `${s.from.toFixed(1)}–${s.to.toFixed(1)} s: ${s.cents > 0 ? "sharp" : "flat"} by about ${semis.toFixed(1)} semitone${semis >= 1.05 ? "s" : ""}`;
      ul.appendChild(li);
    }
    box.append(h, ul);
  } else {
    box.innerHTML = "<p class='hint' style='margin-top:12px'>No stretch where you were clearly off. Nice.</p>";
  }
  $("meta").textContent = `${r.comparedSeconds.toFixed(1)} s compared out of ${last.duration.toFixed(1)} s · analyzed in ${m.seconds.toFixed(0)} s using ${m.backend === "webgpu" ? "the GPU" : "the CPU"}${level ? ` (light mode ${level})` : ""}`;
  $("results").hidden = false;
  drawChart();
}

// Shift each singer pitch by whole octaves to sit nearest the user's pitch, since octaves are ignored.
function singerNearUser(fu, fs) {
  const out = new Float32Array(fs.length).fill(NaN);
  let lastShift = 0;
  const offs = [];
  for (let i = 0; i < fs.length; i++) if (!isNaN(fs[i]) && !isNaN(fu[i])) offs.push(Math.round((midi(fu[i]) - midi(fs[i])) / 12));
  offs.sort((a, b) => a - b);
  if (offs.length) lastShift = offs[Math.floor(offs.length / 2)];
  for (let i = 0; i < fs.length; i++) {
    if (isNaN(fs[i])) continue;
    const shift = isNaN(fu[i]) ? lastShift : Math.round((midi(fu[i]) - midi(fs[i])) / 12);
    out[i] = midi(fs[i]) + 12 * shift;
  }
  return out;
}

function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }

function drawChart(playT = null) {
  if (!last || !last.r) return;
  const { r, duration } = last;
  const cv = $("chart"), dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = 360;
  cv.style.height = `${H}px`;
  cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr);
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const C = { ink: cssVar("--ink"), muted: cssVar("--muted"), line: cssVar("--line"), you: cssVar("--you"), singer: cssVar("--singer"), good: cssVar("--good"), warn: cssVar("--warn"), bad: cssVar("--bad") };
  const L = 44, R = 10, top1 = 8, h1 = 190, top2 = 222, h2 = 112;
  const n = r.fUser.length, hop = r.hop;
  const x = (t) => L + (t / duration) * (W - L - R);
  g.font = `11px ${cssVar("--mono") || "monospace"}`;

  // top: pitch lanes
  const mu = Array.from(r.fUser, (f) => (isNaN(f) ? NaN : midi(f)));
  const ms = singerNearUser(r.fUser, r.fSinger);
  const vals = mu.concat(Array.from(ms)).filter((v) => !isNaN(v)).sort((a, b) => a - b);
  let lo = vals.length ? vals[Math.floor(vals.length * 0.02)] - 2 : 48, hi = vals.length ? vals[Math.floor(vals.length * 0.98)] + 2 : 72;
  if (hi - lo < 12) { const c = (hi + lo) / 2; lo = c - 6; hi = c + 6; }
  const y1 = (m) => top1 + h1 - ((m - lo) / (hi - lo)) * h1;
  g.strokeStyle = C.line; g.fillStyle = C.muted; g.lineWidth = 1;
  const stepN = hi - lo > 24 ? 12 : hi - lo > 14 ? 4 : 2;
  for (let m = Math.ceil(lo); m <= hi; m++) {
    if (m % stepN && m % 12) continue;
    g.globalAlpha = m % 12 === 0 ? 0.9 : 0.45;
    g.beginPath(); g.moveTo(L, y1(m)); g.lineTo(W - R, y1(m)); g.stroke();
    g.globalAlpha = 1;
    g.fillText(`${NOTE[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`, 4, y1(m) + 4);
  }
  const dots = (arr, color, size) => {
    g.fillStyle = color;
    for (let i = 0; i < n; i++) { const v = arr[i]; if (isNaN(v) || v < lo || v > hi) continue; g.fillRect(x(i * hop) - size / 2, y1(v) - size / 2, size, size); }
  };
  dots(ms, C.singer, 3);
  dots(mu, C.you, 2.5);

  // bottom: gap in cents with bands
  const lim = 400, y2 = (c) => top2 + h2 / 2 - (Math.max(-lim, Math.min(lim, c)) / lim) * (h2 / 2);
  const band = (a, b, color, alpha) => { g.globalAlpha = alpha; g.fillStyle = color; g.fillRect(L, y2(b), W - L - R, y2(a) - y2(b)); g.globalAlpha = 1; };
  band(100, lim, C.bad, 0.12); band(-lim, -100, C.bad, 0.12);
  band(50, 100, C.warn, 0.18); band(-100, -50, C.warn, 0.18);
  band(-50, 50, C.good, 0.2);
  g.fillStyle = C.muted;
  for (const c of [-400, -100, 0, 100, 400]) g.fillText(`${c > 0 ? "+" : ""}${c}`, 4, y2(c) + 4);
  g.fillText("sharp", W - R - 38, top2 + 12); g.fillText("flat", W - R - 30, top2 + h2 - 4);
  g.fillStyle = C.muted; g.globalAlpha = 0.5;
  for (let i = 0; i < n; i++) if (!isNaN(r.dev[i])) g.fillRect(x(i * hop) - 1, y2(r.dev[i]) - 1, 2, 2);
  g.globalAlpha = 1;
  g.strokeStyle = C.ink; g.lineWidth = 2; g.beginPath();
  let pen = false;
  for (let i = 0; i < n; i++) {
    const v = r.smooth[i];
    if (isNaN(v)) { pen = false; continue; }
    if (!pen) { g.moveTo(x(i * hop), y2(v)); pen = true; } else g.lineTo(x(i * hop), y2(v));
  }
  g.stroke();

  // time axis
  g.fillStyle = C.muted;
  const tStep = duration > 40 ? 10 : duration > 16 ? 5 : 2;
  for (let t = 0; t <= duration; t += tStep) g.fillText(`${t}s`, x(t) - 6, H - 4);

  if (playT != null) {
    g.strokeStyle = C.bad; g.lineWidth = 1.5;
    g.beginPath(); g.moveTo(x(playT), top1); g.lineTo(x(playT), top2 + h2); g.stroke();
  }
}
window.addEventListener("resize", () => drawChart());
matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => drawChart());

// ---- playback ---------------------------------------------------------------
function tone(f0, hop, n, sr) {
  const y = new Float32Array(n);
  let ph = 0, env = 0, lastF = 220;
  for (let i = 0; i < n; i++) {
    const k = Math.min(f0.length - 1, Math.floor(i / sr / hop));
    const f = f0[k], on = !isNaN(f);
    if (on) lastF = f;
    env += ((on ? 1 : 0) - env) * 0.002;
    ph += (2 * Math.PI * lastF) / sr;
    y[i] = 0.22 * env * (Math.sin(ph) + 0.4 * Math.sin(2 * ph) + 0.15 * Math.sin(3 * ph)) / 1.55;
  }
  return y;
}

function stopPlayback() {
  if (playing) { try { playing.src.stop(); } catch { /* already stopped */ } playing = null; drawChart(); }
}

function play(kind) {
  stopPlayback();
  if (kind === "stop" || !last || !last.r) return;
  const ac = ctx(), rec = last.recording, n = rec.length;
  const peak = rec.reduce((p, v) => Math.max(p, Math.abs(v)), 1e-9);
  const buf = ac.createBuffer(2, n, SR);
  const Lc = buf.getChannelData(0), Rc = buf.getChannelData(1);
  if (kind === "rec") { for (let i = 0; i < n; i++) Lc[i] = Rc[i] = rec[i] / peak * 0.8; }
  else {
    const you = kind === "you" || kind === "overlay" ? tone(last.r.fUser, last.r.hop, n, SR) : null;
    const sg = kind === "singer" || kind === "overlay" ? tone(last.r.fSinger, last.r.hop, n, SR) : null;
    for (let i = 0; i < n; i++) {
      const bed = kind === "overlay" ? rec[i] / peak * 0.35 : 0;
      Lc[i] = bed + (you ? you[i] : sg[i]);
      Rc[i] = bed + (sg ? sg[i] : you[i]);
    }
  }
  const src = ac.createBufferSource();
  src.buffer = buf; src.connect(ac.destination);
  src.onended = () => { if (playing && playing.src === src) { playing = null; drawChart(); } };
  src.start();
  playing = { src, start: ac.currentTime };
  const step = () => {
    if (!playing || playing.src !== src) return;
    drawChart(ac.currentTime - playing.start);
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}
document.querySelectorAll("[data-play]").forEach((b) => b.addEventListener("click", () => play(b.dataset.play)));

// ---- start -------------------------------------------------------------------
if (!window.isSecureContext || !navigator.mediaDevices) {
  hint.textContent = "Recording needs a secure page (https or localhost). You can still analyze a recording file.";
}
// Cross-origin isolation (via sw.js) lets the analysis use several CPU threads.
async function ensureIsolation() {
  if (window.crossOriginIsolated || !("serviceWorker" in navigator) || !window.isSecureContext) return true;
  try {
    const reg = await navigator.serviceWorker.register("./sw.js");
    if (navigator.serviceWorker.controller) return true;   // controlled but still not isolated: give up quietly
    await navigator.serviceWorker.ready;
    if (!sessionStorage.getItem("tc-reloaded")) { sessionStorage.setItem("tc-reloaded", "1"); location.reload(); return false; }
  } catch { /* fall back to one thread */ }
  return true;
}

if (!window.Worker || !window.OfflineAudioContext) {
  setError("This browser can't run the analysis. Try a recent Safari or Chrome.");
} else {
  ensureIsolation().then((go) => go && startWorker());
}
