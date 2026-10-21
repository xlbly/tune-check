// Compare the user's pitch with the original singer's, frame by frame, and summarize.
// Port of analyze.py (split method): octaves are ignored, deviations are smoothed over
// 0.5 s, and moments where either voice is silent are not scored.

export const IN_TUNE_CENTS = 50, OFF_CENTS = 100, SMOOTH_S = 0.5, SLACK_S = 0.15;

const fold = (fu, fs) => { const d = 1200 * Math.log2(fu / fs); return ((d + 600) % 1200 + 1200) % 1200 - 600; };

export function score(fUser, fSinger, hop) {
  const n = Math.min(fUser.length, fSinger.length);
  const w = Math.round(SLACK_S / hop), h = Math.round(SMOOTH_S / hop / 2);
  const dev = new Float32Array(n).fill(NaN), smooth = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    if (isNaN(fUser[i]) || isNaN(fSinger[i])) continue;
    let best = NaN;                         // closest singer pitch within the timing slack
    for (let j = Math.max(0, i - w); j <= Math.min(n - 1, i + w); j++) {
      if (isNaN(fSinger[j])) continue;
      const d = fold(fUser[i], fSinger[j]);
      if (isNaN(best) || Math.abs(d) < Math.abs(best)) best = d;
    }
    dev[i] = best;
  }
  for (let i = 0; i < n; i++) {
    if (isNaN(dev[i])) continue;
    const seg = [];
    for (let j = Math.max(0, i - h); j <= Math.min(n - 1, i + h); j++) if (!isNaN(dev[j])) seg.push(dev[j]);
    seg.sort((a, b) => a - b);
    const m = seg.length;
    smooth[i] = m % 2 ? seg[(m - 1) / 2] : (seg[m / 2 - 1] + seg[m / 2]) / 2;
  }
  let scored = 0, inTune = 0, slight = 0, off = 0;
  for (let i = 0; i < n; i++) {
    if (isNaN(smooth[i])) continue;
    scored++;
    const a = Math.abs(smooth[i]);
    if (a <= IN_TUNE_CENTS) inTune++; else if (a <= OFF_CENTS) slight++; else off++;
  }
  // stretches of at least 0.2 s beyond OFF_CENTS
  const stretches = [];
  let start = -1;
  for (let i = 0; i <= n; i++) {
    const o = i < n && !isNaN(smooth[i]) && Math.abs(smooth[i]) > OFF_CENTS;
    if (o && start < 0) start = i;
    else if (!o && start >= 0) {
      if ((i - 1 - start) * hop >= 0.2) {
        const vals = Array.from(smooth.subarray(start, i)).filter((v) => !isNaN(v)).sort((a, b) => a - b);
        const med = vals[Math.floor(vals.length / 2)];
        stretches.push({ from: start * hop, to: (i - 1) * hop, cents: Math.round(med) });
      }
      start = -1;
    }
  }
  const pct = (k) => (scored ? (100 * k) / scored : null);
  return {
    dev, smooth, stretches, comparedSeconds: scored * hop,
    inTunePct: pct(inTune), slightlyOffPct: pct(slight), offPct: pct(off),
  };
}
