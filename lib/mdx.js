// MDX-Net source separation, a JavaScript port of audio-separator's MDXSeparator
// (hop 1024, 256-frame segments, 25% overlap, no denoise).

import { stft, istft } from "./fft.js";

export const MODELS = {
  // removes the instruments: output = vocals (both voices)
  vocals: { file: "Kim_Vocal_2.onnx", nFft: 7680, dimF: 3072, compensate: 1.009 },
  // karaoke model: output = everything but the lead voice (here: the original singer);
  // lead = input - output, which is the person singing near the mic
  karaoke: { file: "UVR_MDXNET_KARA_2.onnx", nFft: 5120, dimF: 2048, compensate: 1.065 },
};

const HOP = 1024, OVERLAP = 0.25, NORM_PEAK = 0.9;
// Segment length in STFT frames. The models were trained on 256, but their layers work on
// any multiple of 32; shorter segments cut the working memory (about 4x at 64 frames), which
// iOS Safari needs.
export let DIM_T = 256;
export function setSegmentFrames(n) { DIM_T = n; }

function peakOf(ch) { let p = 0; for (const c of ch) for (let i = 0; i < c.length; i++) p = Math.max(p, Math.abs(c[i])); return p; }
function scaled(ch, g) { return ch.map((c) => { const o = new Float32Array(c.length); for (let i = 0; i < c.length; i++) o[i] = c[i] * g; return o; }); }
function capPeak(ch) { const p = peakOf(ch); return p > NORM_PEAK ? scaled(ch, NORM_PEAK / p) : ch; }

// Run the model on one segment (2 channels x chunk samples). Returns 2 channels x chunk.
async function runSegment(seg, cfg, runModel) {
  const { nFft, dimF } = cfg;
  const input = new Float32Array(4 * dimF * DIM_T);
  for (let ch = 0; ch < 2; ch++) {
    const S = stft(seg[ch], nFft, HOP, dimF);
    for (let t = 0; t < DIM_T; t++) {
      for (let f = 3; f < dimF; f++) {                     // the first 3 bins are zeroed
        input[((ch * 2) * dimF + f) * DIM_T + t] = S.re[t * dimF + f];
        input[((ch * 2 + 1) * dimF + f) * DIM_T + t] = S.im[t * dimF + f];
      }
    }
  }
  const out = await runModel(input, [1, 4, dimF, DIM_T]);
  const res = [];
  for (let ch = 0; ch < 2; ch++) {
    const re = new Float32Array(DIM_T * dimF), im = new Float32Array(DIM_T * dimF);
    for (let t = 0; t < DIM_T; t++) for (let f = 0; f < dimF; f++) {
      re[t * dimF + f] = out[((ch * 2) * dimF + f) * DIM_T + t];
      im[t * dimF + f] = out[((ch * 2 + 1) * dimF + f) * DIM_T + t];
    }
    res.push(istft({ re, im, frames: DIM_T, nBins: dimF }, nFft, HOP));
  }
  return res;
}

async function demix(mix, cfg, runModel, onProgress) {
  const len = mix[0].length, chunk = HOP * (DIM_T - 1), trim = cfg.nFft / 2, gen = chunk - 2 * trim;
  const pad = gen + trim - (len % gen), ML = trim + len + pad;
  const mixture = mix.map((c) => { const m = new Float32Array(ML); m.set(c, trim); return m; });
  const step = Math.floor((1 - OVERLAP) * chunk);
  const result = [new Float64Array(ML), new Float64Array(ML)], divider = new Float64Array(ML);
  const nChunks = Math.ceil(ML / step);
  for (let i = 0, k = 0; i < ML; i += step, k++) {
    const end = Math.min(i + chunk, ML), actual = end - i;
    const seg = mixture.map((m) => { const s = new Float32Array(chunk); s.set(m.subarray(i, end)); return s; });
    const tar = await runSegment(seg, cfg, runModel);
    for (let j = 0; j < actual; j++) {
      const w = actual > 1 ? 0.5 - 0.5 * Math.cos(2 * Math.PI * j / (actual - 1)) : 1;   // np.hanning
      divider[i + j] += w;
      result[0][i + j] += tar[0][j] * w;
      result[1][i + j] += tar[1][j] * w;
    }
    onProgress?.((k + 1) / nChunks);
  }
  return result.map((r) => {
    const o = new Float32Array(len);
    for (let j = 0; j < len; j++) { const d = divider[trim + j]; o[j] = d > 0 ? r[trim + j] / d : 0; }
    return o;
  });
}

// Separate a stereo signal (array of 2 Float32Array at 44.1 kHz).
// Returns { primary, secondary }: the model's output stem and input minus it.
export async function separate(stereo, cfg, runModel, onProgress) {
  const peak = peakOf(stereo);
  const mix = peak > NORM_PEAK ? scaled(stereo, NORM_PEAK / peak) : stereo;
  const source = scaled(await demix(mix, cfg, runModel, onProgress), peak);
  const secondary = mix.map((c, ch) => {
    const o = new Float32Array(c.length);
    for (let i = 0; i < c.length; i++) o[i] = c[i] - source[ch][i] * cfg.compensate;
    return o;
  });
  return { primary: capPeak(source), secondary: capPeak(secondary) };
}
