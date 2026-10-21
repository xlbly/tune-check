// Mixed-radix (2, 3, 4, 5) complex FFT, Stockham autosort, plus the STFT / ISTFT used by
// the separation models (torch.stft / torch.istft with center=True, reflect padding and a
// periodic Hann window).

const plans = new Map();

function factorize(n) {
  const f = [];
  for (const p of [4, 2, 3, 5]) while (n % p === 0) { f.push(p); n /= p; }
  if (n !== 1) throw new Error("FFT size must factor into 2, 3 and 5");
  return f;
}

function plan(n) {
  let p = plans.get(n);
  if (!p) {
    const cos = new Float64Array(n), sin = new Float64Array(n);
    for (let j = 0; j < n; j++) { cos[j] = Math.cos(-2 * Math.PI * j / n); sin[j] = Math.sin(-2 * Math.PI * j / n); }
    p = { n, factors: factorize(n), cos, sin, wr: new Float64Array(n), wi: new Float64Array(n) };
    plans.set(n, p);
  }
  return p;
}

// In-place forward FFT of (re, im), both Float64Array of length n.
export function fft(re, im) {
  const P = plan(re.length), N = P.n, C = P.cos, S = P.sin;
  let xr = re, xi = im, yr = P.wr, yi = P.wi;
  let n = N, s = 1;
  const ar = new Float64Array(5), ai = new Float64Array(5);
  for (const p of P.factors) {
    const m = n / p, step = N / n, pstep = N / p;
    for (let q = 0; q < m; q++) {
      for (let k = 0; k < s; k++) {
        for (let r = 0; r < p; r++) { const i = k + s * (q + r * m); ar[r] = xr[i]; ai[r] = xi[i]; }
        for (let t = 0; t < p; t++) {
          let sr = 0, si = 0;
          for (let r = 0; r < p; r++) {
            const w = ((r * t) % p) * pstep;
            sr += ar[r] * C[w] - ai[r] * S[w];
            si += ar[r] * S[w] + ai[r] * C[w];
          }
          const w = (q * t * step) % N;
          const o = k + s * (p * q + t);
          yr[o] = sr * C[w] - si * S[w];
          yi[o] = sr * S[w] + si * C[w];
        }
      }
    }
    [xr, yr] = [yr, xr]; [xi, yi] = [yi, xi];
    n = m; s *= p;
  }
  if (xr !== re) { re.set(xr); im.set(xi); }
}

const windows = new Map();
function hannPeriodic(n) {
  let w = windows.get(n);
  if (!w) { w = new Float64Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / n); windows.set(n, w); }
  return w;
}

// STFT of x (Float32Array). Returns {re, im} as arrays of frames, keeping the first nBins bins.
export function stft(x, nFft, hop, nBins) {
  const half = nFft / 2, L = x.length, win = hannPeriodic(nFft);
  const frames = 1 + Math.floor(L / hop);
  const re = new Float32Array(frames * nBins), im = new Float32Array(frames * nBins);
  const br = new Float64Array(nFft), bi = new Float64Array(nFft);
  const at = (i) => { // reflect padding
    if (i < 0) i = -i;
    if (i >= L) i = 2 * (L - 1) - i;
    return x[i];
  };
  for (let f = 0; f < frames; f++) {
    const start = f * hop - half;
    for (let j = 0; j < nFft; j++) { br[j] = at(start + j) * win[j]; bi[j] = 0; }
    fft(br, bi);
    for (let b = 0; b < nBins; b++) { re[f * nBins + b] = br[b]; im[f * nBins + b] = bi[b]; }
  }
  return { re, im, frames, nBins };
}

// Inverse of stft: bins above nBins are zero. Output length hop * (frames - 1).
export function istft(spec, nFft, hop) {
  const { re, im, frames, nBins } = spec, half = nFft / 2, win = hannPeriodic(nFft);
  const total = nFft + hop * (frames - 1);
  const out = new Float64Array(total), env = new Float64Array(total);
  const br = new Float64Array(nFft), bi = new Float64Array(nFft);
  for (let f = 0; f < frames; f++) {
    br.fill(0); bi.fill(0);
    for (let b = 0; b < nBins && b <= half; b++) {
      br[b] = re[f * nBins + b]; bi[b] = -im[f * nBins + b];       // conj for inverse via forward FFT
      if (b > 0 && b < half) { br[nFft - b] = re[f * nBins + b]; bi[nFft - b] = im[f * nBins + b]; }
    }
    bi[0] = 0; bi[half] = 0;                                        // irfft ignores imag of DC / Nyquist
    fft(br, bi);
    const o = f * hop;
    for (let j = 0; j < nFft; j++) { out[o + j] += (br[j] / nFft) * win[j]; env[o + j] += win[j] * win[j]; }
  }
  const len = hop * (frames - 1), y = new Float32Array(len);
  for (let i = 0; i < len; i++) { const e = env[i + half]; y[i] = e > 1e-11 ? out[i + half] / e : 0; }
  return y;
}
