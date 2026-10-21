// pYIN pitch tracker, a port of librosa.pyin with its defaults
// (frame 2048, hop 512 at 22.05 kHz, 100 thresholds, beta(2, 18), Boltzmann 2,
//  0.1-semitone bins, max transition 35.92 semitones/s, switch prob 0.01).

import { fft } from "./fft.js";

const SR = 22050, FRAME = 2048, HOP = 512, N_THR = 100, BOLTZ = 2, NO_TROUGH = 0.01;
const RES_BINS = 10, MAX_RATE = 35.92, SWITCH = 0.01, MIN_TRANS = 1e-4;

// Regularized incomplete beta I_x(a, b) for integer a, b.
function betaCdf(x, a, b) {
  const n = a + b - 1;
  let s = 0, c = 1;                        // c = C(n, j)
  for (let j = 0; j <= n; j++) {
    if (j >= a) s += c * Math.pow(x, j) * Math.pow(1 - x, n - j);
    c = c * (n - j) / (j + 1);
  }
  return s;
}

const BETA_PROBS = (() => {
  const p = new Float64Array(N_THR);
  for (let i = 0; i < N_THR; i++) p[i] = betaCdf((i + 1) / N_THR, 2, 18) - betaCdf(i / N_THR, 2, 18);
  return p;
})();

function boltzmannPmf(k, lambda, N) {
  return (1 - Math.exp(-lambda)) * Math.exp(-lambda * k) / (1 - Math.exp(-lambda * N));
}

// Decimate 44.1 kHz to 22.05 kHz (windowed-sinc low-pass at ~10 kHz).
export function halve(x) {
  const taps = 97, h = new Float64Array(taps), c = (taps - 1) / 2, fc = 0.45 / 2 * 2; // cutoff 0.45 * Nyquist_out
  let sum = 0;
  for (let i = 0; i < taps; i++) {
    const t = i - c, sinc = t === 0 ? 1 : Math.sin(Math.PI * fc * t) / (Math.PI * fc * t);
    const w = 0.42 - 0.5 * Math.cos(2 * Math.PI * i / (taps - 1)) + 0.08 * Math.cos(4 * Math.PI * i / (taps - 1));
    h[i] = sinc * w; sum += h[i];
  }
  const n = Math.floor(x.length / 2), y = new Float32Array(n);
  for (let o = 0; o < n; o++) {
    let acc = 0; const center = 2 * o;
    for (let i = 0; i < taps; i++) { const j = center + i - c; if (j >= 0 && j < x.length) acc += x[j] * h[i]; }
    y[o] = acc / sum;
  }
  return y;
}

export function pyin(y, fmin = 70, fmax = 1000) {
  const minP = Math.floor(SR / fmax), maxP = Math.min(Math.ceil(SR / fmin), FRAME - 1);
  const padded = new Float32Array(y.length + FRAME);   // center=True, constant padding
  padded.set(y, FRAME / 2);
  const nFrames = 1 + Math.floor((padded.length - FRAME) / HOP);
  const nYin = maxP - minP + 1;
  const nBins = Math.floor(12 * RES_BINS * Math.log2(fmax / fmin)) + 1;
  const obs = new Float64Array(nFrames * 2 * nBins);
  const N2 = 4096, ar = new Float64Array(N2), ai = new Float64Array(N2);
  const yin = new Float64Array(nYin), diff = new Float64Array(maxP + 1), cum = new Float64Array(FRAME);
  const thresholds = Array.from({ length: N_THR + 1 }, (_, i) => i / N_THR);

  for (let f = 0; f < nFrames; f++) {
    const off = f * HOP;
    // autocorrelation via FFT
    ar.fill(0); ai.fill(0);
    for (let j = 0; j < FRAME; j++) ar[j] = padded[off + j];
    fft(ar, ai);
    for (let j = 0; j < N2; j++) { ar[j] = ar[j] * ar[j] + ai[j] * ai[j]; ai[j] = 0; }
    fft(ar, ai);                                   // real, symmetric -> forward == inverse * N
    let e = 0;
    for (let j = 0; j < FRAME; j++) { const v = padded[off + j]; e += v * v; cum[j] = e; }
    const acf0 = ar[0] / N2;
    diff[0] = 0;
    for (let k = 1; k <= maxP; k++) diff[k] = 2 * (acf0 - ar[k] / N2) - cum[k - 1];
    let run = 0;
    const cm = new Float64Array(maxP + 1);
    for (let k = 1; k <= maxP; k++) { run += diff[k]; cm[k] = run / k; }
    for (let i = 0; i < nYin; i++) { const k = minP + i; yin[i] = diff[k] / (cm[k] + 1e-300); }

    // troughs
    const troughs = [];
    for (let i = 0; i < nYin; i++) {
      let t;
      if (i === 0) t = yin[0] < yin[1];
      else if (i === nYin - 1) t = yin[i] < yin[i - 1];
      else t = yin[i] < yin[i - 1] && yin[i] <= yin[i + 1];
      if (t) troughs.push(i);
    }
    const base = f * 2 * nBins;
    let voicedP = 0;
    if (troughs.length) {
      const heights = troughs.map((i) => yin[i]);
      const probs = new Float64Array(troughs.length);
      for (let th = 0; th < N_THR; th++) {
        const thr = thresholds[th + 1];
        let n = 0; for (const h of heights) if (h < thr) n++;
        let pos = 0;
        for (let ti = 0; ti < troughs.length; ti++) {
          if (heights[ti] < thr) { probs[ti] += boltzmannPmf(pos, BOLTZ, n) * BETA_PROBS[th]; pos++; }
        }
      }
      let gmin = 0; for (let ti = 1; ti < heights.length; ti++) if (heights[ti] < heights[gmin]) gmin = ti;
      let below = 0; for (let th = 0; th < N_THR; th++) if (!(heights[gmin] < thresholds[th + 1])) below += BETA_PROBS[th];
      probs[gmin] += NO_TROUGH * below;
      for (let ti = 0; ti < troughs.length; ti++) {
        if (probs[ti] <= 0) continue;
        const i = troughs[ti];
        let shift = 0;
        if (i > 0 && i < nYin - 1) {
          const a = yin[i + 1] + yin[i - 1] - 2 * yin[i], b = (yin[i + 1] - yin[i - 1]) / 2;
          shift = Math.abs(b) >= Math.abs(a) ? 0 : -b / a;
        }
        const f0 = SR / (minP + i + shift);
        let bin = Math.round(12 * RES_BINS * Math.log2(f0 / fmin));
        bin = Math.min(Math.max(bin, 0), nBins);
        obs[base + bin] = probs[ti];               // like numpy fancy assignment: last write wins
      }
      for (let b = 0; b < nBins; b++) voicedP += obs[base + b];
      voicedP = Math.min(Math.max(voicedP, 0), 1);
    }
    for (let b = nBins; b < 2 * nBins; b++) obs[base + b] = (1 - voicedP) / nBins;
  }

  // transition: triangular band within voicing state, times switch matrix
  const maxSemis = Math.round(MAX_RATE * 12 * HOP / SR), width = maxSemis * RES_BINS + 1, hw = Math.floor(width / 2);
  const tri = new Float64Array(width);
  for (let n = 0; n < width; n++) tri[n] = 1 - Math.abs(n - (width - 1) / 2) / ((width + 1) / 2);
  const rowNorm = new Float64Array(nBins);
  for (let i = 0; i < nBins; i++) {
    let s = 0; for (let d = -hw; d <= hw; d++) { const j = i + d; if (j >= 0 && j < nBins) s += tri[d + hw]; }
    rowNorm[i] = s;
  }
  const S2 = 2 * nBins, logMin = Math.log(MIN_TRANS + 1e-300);
  const tiny = 2.2250738585072014e-308;
  // log transition probabilities, indexed by [band offset][source bin], same state / switch
  const lpSame = new Float64Array(width * nBins), lpSwitch = new Float64Array(width * nBins);
  for (let k = 0; k < width; k++) for (let ib = 0; ib < nBins; ib++) {
    const tw = tri[k] / rowNorm[ib];
    lpSame[k * nBins + ib] = Math.log(tw * (1 - SWITCH) + tiny);
    lpSwitch[k * nBins + ib] = Math.log(tw * SWITCH + tiny);
  }
  let value = new Float64Array(S2), next = new Float64Array(S2);
  const ptr = new Uint16Array(nFrames * S2);
  const logInit = Math.log(1 / S2 + tiny);
  for (let s = 0; s < S2; s++) value[s] = Math.log(obs[s] + tiny) + logInit;
  for (let f = 1; f < nFrames; f++) {
    for (let j = 0; j < S2; j++) {
      const jb = j % nBins, jv = j < nBins;
      let best = -Infinity, arg = 0;
      for (let d = -hw; d <= hw; d++) {
        const ib = jb + d;
        if (ib < 0 || ib >= nBins) continue;
        const t = (d + hw) * nBins + ib;
        const ls = lpSame[t], lw = lpSwitch[t];
        const iSame = jv ? ib : ib + nBins, iSw = jv ? ib + nBins : ib;
        if (ls > logMin) { const v = value[iSame] + ls; if (v > best) { best = v; arg = iSame; } }
        if (lw > logMin) { const v = value[iSw] + lw; if (v > best) { best = v; arg = iSw; } }
      }
      next[j] = best + Math.log(obs[f * S2 + j] + tiny);
      ptr[f * S2 + j] = arg;
    }
    [value, next] = [next, value];
  }
  const states = new Int32Array(nFrames);
  let bs = 0; for (let s = 1; s < S2; s++) if (value[s] > value[bs]) bs = s;
  states[nFrames - 1] = bs;
  for (let f = nFrames - 1; f > 0; f--) states[f - 1] = ptr[f * S2 + states[f]];
  const f0 = new Float32Array(nFrames);
  for (let f = 0; f < nFrames; f++) {
    const s = states[f];
    f0[f] = s < nBins ? fmin * Math.pow(2, s / (12 * RES_BINS)) : NaN;
  }
  return { f0, hopSeconds: HOP / SR };
}
