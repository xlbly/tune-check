// Full analysis: 44.1 kHz stereo in, pitch tracks and score out.
import { MODELS, separate } from "./mdx.js";
import { pyin, halve } from "./pyin.js";
import { score } from "./score.js";

const mono = (st) => { const o = new Float32Array(st[0].length); for (let i = 0; i < o.length; i++) o[i] = (st[0][i] + st[1][i]) / 2; return o; };

// getRunner(cfg) -> async (Float32Array, dims) => Float32Array
// onStep(label, fraction) reports progress.
export async function analyze(stereo, getRunner, onStep = () => {}) {
  // one model in memory at a time: each runner may carry a release() that frees it
  onStep("Removing the instruments", 0);
  const r1 = await getRunner(MODELS.vocals);
  const v = await separate(stereo, MODELS.vocals, r1, (p) => onStep("Removing the instruments", p));
  await r1.release?.();
  onStep("Separating your voice from the singer's", 0);
  const r2 = await getRunner(MODELS.karaoke);
  const k = await separate(v.primary, MODELS.karaoke, r2, (p) => onStep("Separating your voice from the singer's", p));
  await r2.release?.();
  const user = mono(k.secondary), singer = mono(k.primary);
  onStep("Tracking pitch", 0);
  await new Promise((r) => setTimeout(r, 0));
  const pu = pyin(halve(user));
  onStep("Tracking pitch", 0.5);
  await new Promise((r) => setTimeout(r, 0));
  const ps = pyin(halve(singer));
  const result = score(pu.f0, ps.f0, pu.hopSeconds);
  return { ...result, fUser: pu.f0, fSinger: ps.f0, hop: pu.hopSeconds, userAudio: user, singerAudio: singer };
}
