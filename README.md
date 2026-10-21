# Tune Check (web app, runs entirely on the device)

Record a snippet while singing along to a song playing on another speaker; the page
separates your voice from the original singer and shows, over time, how in tune you were.
Nothing is uploaded: the models run in the browser (WebGPU when available, otherwise CPU).

## Try it on a computer
    cd tune-web
    python3 -m http.server 8000
Open http://localhost:8000 (the microphone works on localhost). The first visit downloads
the models (~120 MB) and keeps them in the browser cache.

## Open it on a phone
Phones only allow the microphone on https pages, so the folder has to be hosted. Any static
host works, for example GitHub Pages or Netlify Drop (drag the `tune-web` folder onto
https://app.netlify.com/drop). No server code or special headers are needed: `sw.js`
adds the headers that let the analysis use several CPU threads.

## What's inside
- `index.html`, `app.js`: page, recording (MediaRecorder), chart, playback.
- `worker.js`: loads the models with ONNX Runtime Web and runs the analysis off the main thread.
- `lib/mdx.js`, `lib/fft.js`: port of audio-separator's MDX-Net inference (STFT, chunking).
- `lib/pyin.js`: port of librosa's pYIN pitch tracker. `lib/score.js`: scoring, as in `tune-poc/analyze.py`.
- `models/`: Kim_Vocal_2 (removes instruments) and UVR_MDXNET_KARA_2 (splits lead = you from
  backing = original singer), from the UVR model repository. Check their licences before any
  commercial use.
- `vendor/`: ONNX Runtime Web 1.20.1 (MIT).

## Checked
- Separation output matches the Python pipeline exactly (correlation 1.0 on take 5).
- Pitch tracks match librosa within 20 cents on >99% of frames.
- Scores (Node, same code): take 4 = 10% off, take 5 = 42% off (Python: 8% / 41%).
- Full page in headless Chromium with take 5 played into a fake microphone: 40% off,
  same off stretches; 75 s for a 14 s snippet on 4 CPU threads (WebGPU untested).
