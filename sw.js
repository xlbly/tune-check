// Adds cross-origin isolation headers so the analysis can use several CPU threads
// (static hosts like GitHub Pages can't set these headers themselves).
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(fetch(req).then((res) => {
    if (res.status === 0) return res;
    const h = new Headers(res.headers);
    h.set("Cross-Origin-Embedder-Policy", "require-corp");
    h.set("Cross-Origin-Opener-Policy", "same-origin");
    h.set("Cross-Origin-Resource-Policy", "same-origin");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});
