import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* Glide DLSS5
 *
 * Drop a video on the node, scrub to a frame, press Preview: the server
 * renders a short DLSS 5 burst ending at that frame (no queue) and the view
 * turns into a before/after split - move the mouse across it to slide the
 * divider. Queue renders the whole file.
 *
 * The scrubber shows frames the SERVER decodes, not a <video> element: the
 * H3 chain's 4:4:4 HEVC in MKV is not something a browser will play, and a
 * preview tool that cannot show your own clips is no preview tool.
 */

const NODES = ["CSGlideDLSS5CS", "CSGlideDLSS5"];   // current id, then the standalone one
const VIDEO_EXTS = [".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mts", ".m2ts", ".ts"];

const CSS = `
.gd5 { display:flex; flex-direction:column; gap:6px; width:100%; height:100%;
  box-sizing:border-box; font:12px/1.35 system-ui, sans-serif; color:#d8d8d8; }
.gd5-view { position:relative; flex:1 1 auto; min-height:160px; background:#0d0d0d;
  border:1px solid #ffffff1f; border-radius:6px; overflow:hidden; }
.gd5-view.drag { border-color:#4fc3ff; box-shadow:inset 0 0 0 2px #4fc3ff66; }
.gd5-view img { position:absolute; inset:0; width:100%; height:100%; object-fit:contain;
  user-select:none; -webkit-user-drag:none; pointer-events:none; }
.gd5-after { display:none; }
.gd5-view.cmp .gd5-after { display:block; }
.gd5-split { display:none; position:absolute; top:0; bottom:0; width:2px; margin-left:-1px;
  background:#fff; box-shadow:0 0 0 1px #0008, 0 0 8px #fff8; pointer-events:none; }
.gd5-view.cmp .gd5-split { display:block; }
.gd5-tag { display:none; position:absolute; top:6px; padding:1px 6px; border-radius:4px;
  background:#000a; font-size:11px; pointer-events:none; }
.gd5-view.cmp .gd5-tag { display:block; }
.gd5-tag.l { left:6px; } .gd5-tag.r { right:6px; color:#4fc3ff; }
.gd5-hint { position:absolute; inset:0; display:grid; place-items:center; text-align:center;
  color:#8a8a8a; pointer-events:none; padding:12px; }
.gd5-view.has .gd5-hint { display:none; }
.gd5-row { display:flex; align-items:center; gap:6px; }
.gd5-track { position:relative; flex:1 1 auto; min-width:60px; height:18px; cursor:pointer;
  touch-action:none; }
.gd5-track::before { content:""; position:absolute; left:0; right:0; top:8px; height:3px;
  border-radius:2px; background:#ffffff26; }
.gd5-fill { position:absolute; left:0; top:8px; height:3px; border-radius:2px; background:#4fc3ff;
  pointer-events:none; }
.gd5-knob { position:absolute; top:3px; width:12px; height:12px; margin-left:-6px; border-radius:50%;
  background:#fff; box-shadow:0 0 0 2px #4fc3ff; pointer-events:none; }
.gd5-btn { padding:4px 10px; border-radius:5px; border:1px solid #ffffff2e; background:#2a2a2a;
  color:#e6e6e6; cursor:pointer; font:inherit; white-space:nowrap; }
.gd5-btn:hover { border-color:#4fc3ff; }
.gd5-btn.go { background:#1f6f9a; border-color:#4fc3ff; color:#fff; font-weight:600; }
.gd5-btn:disabled { opacity:.45; cursor:default; }
.gd5-read { font-family:ui-monospace, Consolas, monospace; font-size:11px; color:#9a9a9a;
  white-space:nowrap; }
.gd5-status { font-size:11px; color:#9a9a9a; min-height:15px; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.gd5-status.err { color:#ff7a7a; } .gd5-status.ok { color:#7fd48a; }
`;

function injectCss() {
  if (document.getElementById("gd5-css")) return;
  const s = document.createElement("style");
  s.id = "gd5-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
};

const isVideo = (name) => VIDEO_EXTS.some((x) => String(name || "").toLowerCase().endsWith(x));

function getLink(graph, id) {
  if (id == null || !graph) return null;
  const links = graph.links;
  return (links && typeof links.get === "function") ? links.get(id) : links?.[id];
}

/* The connected DLSS5 Settings node's CURRENT widget values - what Preview
 * sends. Follows plain reroutes; anything more exotic (a subgraph boundary)
 * gets a clear message instead of a guess. */
function settingsValues(node) {
  const input = node.inputs?.find((i) => i.name === "settings");
  let link = getLink(node.graph, input?.link);
  if (!link) throw new Error("Connect a DLSS5 Settings node to 'settings' first.");
  let source = node.graph.getNodeById(link.origin_id);
  let hops = 0;
  while (source && /reroute/i.test(source.type || "") && hops++ < 16) {
    const next = getLink(node.graph, source.inputs?.[0]?.link);
    if (!next) break;
    source = node.graph.getNodeById(next.origin_id);
  }
  if (!source || source.type !== "DLSS5Settings") {
    throw new Error("Preview needs a DLSS5 Settings node connected directly (or through reroutes).");
  }
  const values = {};
  for (const w of source.widgets || []) values[w.name] = w.value;
  return values;
}

function viewUrl(ref) {
  const q = new URLSearchParams({ filename: ref.filename, subfolder: ref.subfolder || "",
                                  type: ref.type || "temp", t: String(Date.now()) });
  return api.apiURL(`/view?${q}`);
}

function setup(node) {
  injectCss();
  const vw = node.widgets?.find((w) => w.name === "video");
  const nw = node.widgets?.find((w) => w.name === "preview_frames");

  const root = el("div", "gd5");
  const view = el("div", "gd5-view");
  const imgSrc = el("img");
  const imgAfter = el("img", "gd5-after");
  const split = el("div", "gd5-split");
  const tagL = el("div", "gd5-tag l", "Source");
  const tagR = el("div", "gd5-tag r", "DLSS5");
  const hint = el("div", "gd5-hint", "Drop a video here");
  hint.style.whiteSpace = "pre-line";
  view.append(imgSrc, imgAfter, split, tagL, tagR, hint);

  const row = el("div", "gd5-row");
  const prev = el("button", "gd5-btn", "◀");
  // A drawn track, not <input type=range>: the new frontend takes pointer
  // input at window level, so a native slider drags the canvas instead.
  const range = el("div", "gd5-track");
  const fill = el("div", "gd5-fill");
  const knob = el("div", "gd5-knob");
  range.append(fill, knob);
  const next = el("button", "gd5-btn", "▶");
  const read = el("span", "gd5-read", "–");
  const go = el("button", "gd5-btn go", "Preview");
  row.append(prev, range, next, read, go);

  const paintTrack = () => {
    const max = state.info ? Math.max(1, state.info.frames - 1) : 1;
    const pct = `${(state.frame / max) * 100}%`;
    fill.style.width = pct; knob.style.left = pct;
  };
  const status = el("div", "gd5-status");
  root.append(view, row, status);

  // Keep the graph canvas from treating presses inside the widget as its own
  // (panning, dragging the node) - the controls here want them.
  for (const t of ["pointerdown", "mousedown", "wheel"]) {
    root.addEventListener(t, (e) => e.stopPropagation());
  }

  const state = node.__gd5 = { info: null, frame: 0, file: "", previewFrame: -1, busy: false,
                               pending: 0, split: 0.5 };

  const say = (text, kind = "") => { status.textContent = text; status.className = "gd5-status" + (kind ? " " + kind : ""); };

  const readout = () => {
    const i = state.info;
    if (!i) { read.textContent = "–"; return; }
    read.textContent = `f${state.frame}/${i.frames - 1}  ${(state.frame / i.fps).toFixed(2)}s`;
  };

  const showCompare = (on) => {
    view.classList.toggle("cmp", on);
    if (on) {
      imgAfter.style.clipPath = `inset(0 0 0 ${state.split * 100}%)`;
      split.style.left = `${state.split * 100}%`;
    }
  };

  // Only the latest scrub position is fetched; stale responses are dropped.
  let timer = 0;
  const showFrame = (index) => {
    if (!state.info) return;
    state.frame = Math.max(0, Math.min(state.info.frames - 1, index | 0));
    paintTrack();
    readout();
    if (state.frame !== state.previewFrame) showCompare(false);
    clearTimeout(timer);
    timer = setTimeout(() => {
      const ticket = ++state.pending;
      const q = new URLSearchParams({ file: state.file, index: String(state.frame), w: "1280" });
      const img = new Image();
      img.onload = () => { if (ticket === state.pending) imgSrc.src = img.src; };
      img.src = api.apiURL(`/csglide_dlss5/frame?${q}`);
    }, 50);
  };

  const load = async () => {
    const file = String(vw?.value || "").trim();
    state.file = file; state.info = null; state.previewFrame = -1;
    showCompare(false);
    view.classList.toggle("has", false);
    imgSrc.removeAttribute("src");
    if (!file) { readout(); say(""); return; }
    try {
      const r = await api.fetchApi(`/csglide_dlss5/info?${new URLSearchParams({ file })}`);
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || r.statusText);
      if (file !== state.file) return;
      state.info = j;
      view.classList.add("has");
      say(`${j.width}×${j.height} · ${j.fps.toFixed(3).replace(/\.?0+$/, "")} fps · ${j.frames} frames`
          + (j.pix_fmt ? ` · ${j.pix_fmt}` : ""));
      showFrame(Math.min(state.frame, j.frames - 1));
    } catch (err) {
      say(String(err.message || err), "err");
    }
  };
  state.load = load;

  const upload = async (file) => {
    if (!isVideo(file.name)) { say(`${file.name} is not a video file.`, "err"); return; }
    say(`Uploading ${file.name}…`);
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("subfolder", "glide_dlss5");
    body.append("type", "input");
    try {
      const r = await api.fetchApi("/upload/image", { method: "POST", body });
      const j = await r.json();
      if (!r.ok) throw new Error(j?.error || r.statusText);
      const value = (j.subfolder ? `${j.subfolder}/` : "") + j.name;
      if (vw) { vw.value = value; vw.callback?.(value); }
      state.frame = 0;
      await load();
    } catch (err) {
      say(`Upload failed: ${err.message || err}`, "err");
    }
  };
  state.upload = upload;

  const preview = async () => {
    if (state.busy) return;
    if (!state.info) { say("Drop a video first.", "err"); return; }
    let settings;
    try { settings = settingsValues(node); } catch (err) { say(err.message, "err"); return; }
    state.busy = true; go.disabled = true; go.textContent = "Rendering…";
    const count = Number(nw?.value ?? 8) || 8;
    say(`Rendering ${Math.min(count, state.frame + 1)} frames up to f${state.frame}…`);
    try {
      const r = await api.fetchApi("/csglide_dlss5/preview", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file: state.file, frame: state.frame, count, settings }),
      });
      const j = await r.json();
      if (!r.ok || j.error) throw new Error(j.error || r.statusText);
      await Promise.all([[imgSrc, j.before], [imgAfter, j.after]].map(([img, ref]) =>
        new Promise((res) => { img.onload = img.onerror = () => { img.onload = img.onerror = null; res(); }; img.src = viewUrl(ref); })));
      state.previewFrame = j.frame;
      if (state.frame === j.frame) showCompare(true);
      say(`f${j.frame} · ${j.burst} frames in ${j.seconds}s · ${j.input[0]}×${j.input[1]} → ${j.output[0]}×${j.output[1]} · move the mouse across the image`, "ok");
    } catch (err) {
      say(`Preview failed: ${err.message || err}`, "err");
    } finally {
      state.busy = false; go.disabled = false; go.textContent = "Preview";
    }
  };

  // Hover slides the divider - no drag, so nothing to fight the canvas over.
  view.addEventListener("pointermove", (e) => {
    if (!view.classList.contains("cmp")) return;
    const box = view.getBoundingClientRect();
    state.split = Math.max(0, Math.min(1, (e.clientX - box.left) / Math.max(1, box.width)));
    showCompare(true);
  });

  // Scrub: claimed in the window CAPTURE phase so the canvas never sees it.
  let scrubbing = false;
  const frameAt = (x) => {
    const box = range.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (x - box.left) / Math.max(1, box.width)));
    return Math.round(t * Math.max(0, (state.info?.frames || 1) - 1));
  };
  const onDown = (e) => {
    if (e.button !== 0 || !state.info || !range.contains(e.target)) return;
    e.preventDefault(); e.stopPropagation();
    scrubbing = true;
    showFrame(frameAt(e.clientX));
  };
  const onMove = (e) => {
    if (!scrubbing) return;
    e.preventDefault(); e.stopPropagation();
    showFrame(frameAt(e.clientX));
  };
  const onUp = (e) => {
    if (!scrubbing) return;
    scrubbing = false;
    e.stopPropagation();
  };
  window.addEventListener("pointerdown", onDown, true);
  window.addEventListener("pointermove", onMove, true);
  window.addEventListener("pointerup", onUp, true);
  state.detach = () => {
    window.removeEventListener("pointerdown", onDown, true);
    window.removeEventListener("pointermove", onMove, true);
    window.removeEventListener("pointerup", onUp, true);
  };
  prev.addEventListener("click", () => showFrame(state.frame - 1));
  next.addEventListener("click", () => showFrame(state.frame + 1));
  go.addEventListener("click", preview);

  // Drop straight onto the widget as well as the node: the Vue node renderer
  // does not route LiteGraph's onDragDrop, the DOM always sees this.
  view.addEventListener("dragover", (e) => {
    if (!e.dataTransfer?.types?.includes("Files")) return;
    e.preventDefault(); e.stopPropagation(); view.classList.add("drag");
  });
  view.addEventListener("dragleave", () => view.classList.remove("drag"));
  view.addEventListener("drop", (e) => {
    view.classList.remove("drag");
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    e.preventDefault(); e.stopPropagation();
    upload(file);
  });

  if (vw) {
    const cb = vw.callback;
    vw.callback = function (value) {
      const r = cb?.apply(this, arguments);
      if (String(value || "").trim() !== state.file) { state.frame = 0; load(); }
      return r;
    };
  }

  node.addDOMWidget("glide_dlss5_view", "div", root, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 300,
  });
  if (node.size[0] < 520) node.setSize([520, Math.max(node.size[1], 640)]);
  readout();
}

app.registerExtension({
  name: "CGlide.GlideDLSS5",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (!NODES.includes(nodeData.name)) return;

    const created = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = created?.apply(this, arguments);
      setup(this);
      return r;
    };

    // A saved workflow restores the 'video' value after creation: load it then.
    const configured = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = configured?.apply(this, arguments);
      setTimeout(() => this.__gd5?.load?.(), 0);
      return r;
    };

    const removed = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.__gd5?.detach?.();
      return removed?.apply(this, arguments);
    };

    // Classic canvas: a file dropped anywhere on the node body.
    nodeType.prototype.onDragOver = function (e) {
      return !!e?.dataTransfer?.types?.includes?.("Files");
    };
    nodeType.prototype.onDragDrop = function (e) {
      const file = e?.dataTransfer?.files?.[0];
      if (!file || !isVideo(file.name)) return false;
      this.__gd5?.upload?.(file);
      return true;
    };
  },
  setup() {
    api.addEventListener("executed", ({ detail }) => {
      const done = detail?.output?.glide_dlss5_done?.[0];
      if (!done) return;
      const node = app.graph?.getNodeById?.(Number(detail.node)) ?? app.graph?.getNodeById?.(detail.node);
      const status = node?.__gd5 && node.widgets?.find((w) => w.name === "glide_dlss5_view")?.element?.querySelector?.(".gd5-status");
      if (status) {
        status.textContent = `Rendered ${done.frames} frames → ${done.path}`;
        status.className = "gd5-status ok";
      }
    });
  },
});
