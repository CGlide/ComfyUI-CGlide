import { app } from "../../scripts/app.js";

/*
 * csglide_video.js -- preview widget for the Glide Video node.
 *
 * Renders the encoded clip inline with a visible control bar: play/pause,
 * loop toggle, frame step, jump to first/last frame, and a scrubber.
 * VHS hides all of this behind a right-click menu, which is fine once and
 * annoying the two-hundredth time.
 *
 * Frame step and the frame counter use the fps and frame count the Python
 * side puts in the payload, so they are exact rather than inferred from
 * the video's own timebase.
 */

const CSS = `
.glide-video-wrap {
  display: flex; flex-direction: column; gap: 4px;
  width: 100%; box-sizing: border-box; padding: 2px;
  font-family: system-ui, -apple-system, sans-serif;
}
.glide-video-wrap video {
  width: 100%; display: block; border-radius: 4px;
  background: #000; min-height: 40px;
}
.glide-video-empty {
  display: flex; align-items: center; justify-content: center;
  min-height: 60px; border: 1px dashed #444; border-radius: 4px;
  color: #666; font-size: 11px;
}
.glide-video-bar {
  display: flex; align-items: center; gap: 3px;
  background: #1e1e1e; border-radius: 4px; padding: 3px 5px;
}
.glide-video-bar button {
  background: none; border: none; color: #ccc; cursor: pointer;
  font-size: 12px; line-height: 1; padding: 3px 5px; border-radius: 3px;
  min-width: 22px;
}
.glide-video-bar button:hover { background: #333; color: #fff; }
.glide-video-bar button.on { color: #4fff8f; }
.glide-video-seek {
  flex: 1; min-width: 30px; height: 4px; margin: 0 4px;
  accent-color: #4fff8f; cursor: pointer;
}
.glide-video-time {
  color: #888; font-size: 10px; font-variant-numeric: tabular-nums;
  white-space: nowrap; min-width: 92px; text-align: right;
}
.glide-video-meta {
  color: #666; font-size: 10px; display: flex;
  justify-content: space-between; padding: 0 3px;
}
.glide-video-meta a { color: #4fff8f; text-decoration: none; }
.glide-video-stream {
  color: #4fff8f; font-size: 10px; font-family: ui-monospace, Consolas, monospace;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  padding: 0 6px; flex: 1; text-align: center;
}
.glide-video-meta a:hover { text-decoration: underline; }
.glide-video-detail {
  color: #8a8a8a; font-size: 10px; line-height: 1.45;
  background: #1a1a1a; border-left: 2px solid #4fff8f;
  border-radius: 3px; padding: 5px 7px; margin: 0 2px;
}
.glide-video-detail b { color: #ccc; font-weight: 600; }
`;

/* Detail shown under the preset dropdown. Kept here rather than pulled
 * from Python because it is UI copy, not encoder configuration -- the
 * authoritative settings live in csglide_video_presets.py. */
const PRESET_DETAIL = {
  "H.264 (compatible)":
    "<b>crf 15 &middot; capped &middot; yuv420p &middot; x264 medium</b><br>" +
    "Visually lossless and plays on everything &mdash; phones, browsers, " +
    "every editor. Bitrate follows the content, so a simple shot makes a " +
    "small file. Largest of the share presets. The default for a reason.",
  "H.265 (smaller)":
    "<b>crf 19 &middot; capped &middot; yuv420p &middot; hvc1 tag</b><br>" +
    "Roughly 35% smaller than H.264 at the same look. Deblocking is " +
    "dialled back and SAO disabled so grain and fine texture survive " +
    "&mdash; untuned HEVC softens exactly the detail you prompted for.",
  "AV1 (small, best quality)":
    "<b>cq 24 &middot; capped &middot; 10-bit &middot; NVENC</b><br>" +
    "Smallest share preset, encodes in seconds on the GPU. 10-bit is not " +
    "a mistake: it makes files smaller <i>and</i> kills banding in skies " +
    "and gradients. Falls back to CPU AV1 if NVENC is missing.",
  "ProRes 422 HQ (master)":
    "<b>4:2:2 10-bit &middot; all-intra &middot; 16-bit pipe</b><br>" +
    "For editing and for chunk handoff. Every frame is independent, so " +
    "pulling a single frame is exact and re-encoding costs nothing. " +
    "Large files. Writes a small H.264 alongside for this preview.",
  "H.264 4:4:4 10-bit":
    "<b>crf 12 &middot; yuv444p10le &middot; 16-bit pipe</b><br>" +
    "Keeps all the colour. 4:2:0 throws away three quarters of the " +
    "chroma before the encoder starts, and no bitrate gets it back &mdash; " +
    "the damage lands on saturated edges. Use this for chunk handoff, " +
    "where the last frame becomes the next clip's first. Editors and VLC " +
    "play it; phones and browsers do not.",
  "H.265 4:4:4 10-bit":
    "<b>crf 16 &middot; yuv444p10le &middot; 16-bit pipe &middot; mp4</b><br>" +
    "Same fidelity as the H.264 4:4:4 preset, appreciably smaller. " +
    "Grain-preserving tuning applied. Same playback caveat &mdash; and " +
    "note the container widget rewraps it without touching the codec, " +
    "so a player that chokes on this will choke on the mp4 too.",
  "FFV1 (lossless archive)":
    "<b>RGB 16-bit &middot; mathematically lossless</b><br>" +
    "Bit-exact &mdash; verified round-trip, zero error. Stays in RGB, so " +
    "there is no colour matrix conversion to lose anything to. Very " +
    "large. Writes a small H.264 alongside for this preview.",
};

function injectCSS() {
  if (document.getElementById("glide-video-css")) return;
  const s = document.createElement("style");
  s.id = "glide-video-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

function viewURL(info) {
  const p = new URLSearchParams({
    filename: info.filename,
    subfolder: info.subfolder || "",
    type: info.type || "output",
    // defeat the browser cache when the same filename is regenerated
    r: Math.random().toString(36).slice(2),
  });
  return `/view?${p.toString()}`;
}

function build(node) {
  injectCSS();

  const wrap = document.createElement("div");
  wrap.className = "glide-video-wrap";

  const empty = document.createElement("div");
  empty.className = "glide-video-empty";
  empty.textContent = "no clip yet";
  wrap.appendChild(empty);

  const video = document.createElement("video");
  video.muted = false;
  video.playsInline = true;
  video.preload = "auto";
  video.style.display = "none";
  wrap.appendChild(video);

  const bar = document.createElement("div");
  bar.className = "glide-video-bar";
  bar.style.display = "none";
  wrap.appendChild(bar);

  const mk = (label, title) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.title = title;
    bar.appendChild(b);
    return b;
  };

  const bFirst = mk("|\u25C0", "First frame");
  const bPrev  = mk("\u25C0", "Previous frame");
  const bPlay  = mk("\u25B6", "Play / pause  (space)");
  const bNext  = mk("\u25B6|", "Next frame");
  const bLast  = mk("\u25B6\u25B6|", "Last frame  (the one to hand off)");

  const seek = document.createElement("input");
  seek.type = "range";
  seek.className = "glide-video-seek";
  seek.min = "0"; seek.max = "1000"; seek.value = "0";
  bar.appendChild(seek);

  const time = document.createElement("span");
  time.className = "glide-video-time";
  time.textContent = "0 / 0";
  bar.appendChild(time);

  const bLoop = mk("\u21BB", "Loop");

  const detail = document.createElement("div");
  detail.className = "glide-video-detail";
  wrap.appendChild(detail);

  const meta = document.createElement("div");
  meta.className = "glide-video-meta";
  const metaLeft = document.createElement("span");
  const metaStream = document.createElement("span");
  metaStream.className = "glide-video-stream";
  metaStream.title = "Read back from the finished file, not from the preset";
  const metaRight = document.createElement("a");
  metaRight.textContent = "open";
  metaRight.target = "_blank";
  meta.appendChild(metaLeft);
  meta.appendChild(metaStream);
  meta.appendChild(metaRight);
  meta.style.display = "none";
  wrap.appendChild(meta);

  // ---- state -------------------------------------------------------
  let fps = 24;
  let frames = 0;
  let dragging = false;
  let dims = "";

  const frameDur = () => 1 / (fps || 24);
  const curFrame = () => Math.round(video.currentTime * fps);

  function paintLoop() {
    const on = !!node.properties.glide_loop;
    video.loop = on;
    bLoop.classList.toggle("on", on);
  }

  /* seconds under a minute, m:ss.hh over it */
  function fmtTime(t) {
    if (!isFinite(t) || t < 0) t = 0;
    if (t < 60) return `${t.toFixed(2)}s`;
    const m = Math.floor(t / 60);
    return `${m}:${(t - m * 60).toFixed(2).padStart(5, "0")}`;
  }

  /* dims and the probed stream share one line, dot-separated, either half
     optional -- an old saved payload has no width, and probe_video returns
     nothing when ffprobe is missing. */
  function paintStream(stream) {
    metaStream.textContent = [dims, stream].filter(Boolean).join("  \u00B7  ");
  }

  function paintTime() {
    const dur = video.duration || (frames && fps ? frames / fps : 0);
    time.textContent = `${fmtTime(video.currentTime)} / ${fmtTime(dur)}`;
    time.title = `frame ${curFrame()} of ${frames || "?"}`;
    if (!dragging && video.duration) {
      seek.value = String((video.currentTime / video.duration) * 1000);
    }
  }

  function step(n) {
    video.pause();
    const d = video.duration || 0;
    video.currentTime = Math.max(0, Math.min(d - frameDur() / 2,
                                             video.currentTime + n * frameDur()));
  }

  bFirst.onclick = () => { video.pause(); video.currentTime = 0; };
  bPrev.onclick  = () => step(-1);
  bNext.onclick  = () => step(1);
  bLast.onclick  = () => {
    video.pause();
    if (video.duration) video.currentTime = Math.max(0, video.duration - frameDur());
  };
  bPlay.onclick  = () => { video.paused ? video.play() : video.pause(); };
  bLoop.onclick  = () => {
    node.properties.glide_loop = !node.properties.glide_loop;
    paintLoop();
  };

  seek.oninput = () => {
    dragging = true;
    if (video.duration) video.currentTime = (seek.value / 1000) * video.duration;
  };
  seek.onchange = () => { dragging = false; };

  video.addEventListener("play",  () => { bPlay.textContent = "\u23F8"; });
  video.addEventListener("pause", () => { bPlay.textContent = "\u25B6"; });
  video.addEventListener("timeupdate", paintTime);
  video.addEventListener("seeked", paintTime);
  video.addEventListener("loadedmetadata", () => {
    if (!frames && video.duration) frames = Math.round(video.duration * fps);
    /* Last resort for payloads that predate the width/height fields. */
    if (!dims && video.videoWidth) {
      dims = `${video.videoWidth}\u00D7${video.videoHeight}`;
      paintStream(metaStream.textContent);
    }
    paintTime();
    node.setDirtyCanvas(true, true);
  });
  video.addEventListener("error", () => {
    empty.textContent = "cannot play this format in the browser";
    empty.style.display = "flex";
    video.style.display = "none";
  });

  // ---- public -------------------------------------------------------
  function show(info) {
    if (!info || !info.filename) return;
    fps = info.fps || 24;
    frames = info.frames || 0;

    const url = viewURL(info);
    video.src = url;
    metaRight.href = url;
    metaLeft.textContent = info.master || info.filename;

    /* Resolution first: it is the thing you check at a glance, and unlike
       codec and pix_fmt it is not in the filename either. Comes from the
       Python side so it is right even when the browser cannot decode the
       master at all. */
    dims = (info.width && info.height) ? `${info.width}\u00D7${info.height}` : "";
    paintStream(info.stream || "");

    empty.style.display = "none";
    video.style.display = "block";
    bar.style.display = "flex";
    meta.style.display = "flex";

    paintLoop();
    // remember it so the preview survives a workflow reload
    node.properties.glide_last = info;
  }

  function setDetail(presetName) {
    const html = PRESET_DETAIL[presetName];
    detail.innerHTML = html || "";
    detail.style.display = html ? "block" : "none";
  }

  paintLoop();
  return { wrap, show, setDetail };
}

app.registerExtension({
  name: "CGlide.GlideVideo",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CSGlideVideoCS") return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);

      this.properties = this.properties || {};
      if (this.properties.glide_loop === undefined) {
        this.properties.glide_loop = true;
      }

      const ui = build(this);
      this._glideVideo = ui;
      this.addDOMWidget("glide_preview", "preview", ui.wrap, {
        serialize: false,
        hideOnZoom: false,
      });

      // keep the detail line in step with the preset dropdown
      setTimeout(() => {
        const w = this.widgets?.find((x) => x.name === "preset");
        if (!w) return;
        ui.setDetail(w.value);
        const prev = w.callback;
        w.callback = function (v) {
          const out = prev?.apply(this, arguments);
          ui.setDetail(v);
          return out;
        };
      }, 0);

      this.size = this.computeSize();
      if (this.size[1] < 340) this.size[1] = 340;
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const payload = message?.glide_video;
      const info = Array.isArray(payload) ? payload[0] : payload;
      if (info) this._glideVideo?.show(info);
    };

    // restore the last preview when a saved workflow is reopened
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      const last = this.properties?.glide_last;
      if (last) {
        setTimeout(() => this._glideVideo?.show(last), 0);
      }
      return r;
    };
  },
});
