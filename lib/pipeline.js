// Full analysis: 44.1 kHz stereo in, pitch tracks and score out.
import { MODELS, separate } from "./mdx.js";
import { pyin, halve } from "./pyin.js";
import { score } from "./score.js";

const mono = (st) => { const o = new Float32Array(st[0].length); for (let i = 0; i < o.length; i++) o[i] = (st[0][i] + st[1][i]) / 2; return o; };

// getRunner(cfg) -> async (Float32Array, dims) => Float32Array
// onStep(label, fraction) reports progress.
export async function analyze(stereo, getRunner, onStep = () => {}) {
  onStep("Removing the instruments", 0);
  const v = await separate(stereo, MODELS.vocals, await getRunner(MODELS.vocals), (p) => onStep("Removing the instruments", p));
  onStep("Separating your voice from the singer's", 0);
  const k = await separate(v.primary, MODELS.karaoke, await getRunner(MODELS.karaoke), (p) => onStep("Separating your voice from the singer's", p));
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
