import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/*
 * csglide_preview.js -- the player for Glide Preview.
 *
 * ComfyUI's preview socket encodes one JPEG or PNG per update, so an
 * animation cannot travel that way. The Python side encodes a looping WebP
 * and pushes it over its own event; this just puts it on the node. An
 * animated WebP in an <img> plays by itself, so there is no timer here.
 */

const NODE_ID = "CSGlidePreviewCS";

const CSS = `
.gpv {
  display:flex; flex-direction:column; gap:5px; padding:2px;
  font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  box-sizing:border-box; width:100%;
}
.gpv-stage {
  position:relative; width:100%; background:#0e0e0e;
  border:1px solid #3b3b3b; border-radius:7px; overflow:hidden;
  min-height:90px; display:flex; align-items:center; justify-content:center;
}
.gpv-stage img { width:100%; display:block; image-rendering:auto; }
.gpv-empty { color:#6a6a6a; font-size:11px; padding:26px 10px; text-align:center; }
.gpv-meta {
  display:flex; gap:10px; align-items:center;
  color:#8f8f8f; font-size:10px; font-family:ui-monospace,Consolas,monospace;
}
.gpv-meta b { color:#58d1ff; font-weight:600; }
.gpv-meta .sp { margin-left:auto; }
.gpv-btn { all:unset; pointer-events:auto; cursor:pointer; padding:1px 7px;
  border:1px solid #3b3b3b; border-radius:5px; color:#9a9a9a; font-size:10px; }
.gpv-btn:hover { border-color:#58d1ff; color:#58d1ff; }
.gpv-stage canvas { width:100%; display:block; }
`;

function injectCSS() {
  if (document.getElementById("gpv-css")) return;
  const s = document.createElement("style");
  s.id = "gpv-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

const nodes = new Set();

function build(node) {
  injectCSS();
  const root = document.createElement("div");
  root.className = "gpv";

  const stage = document.createElement("div");
  stage.className = "gpv-stage";
  const empty = document.createElement("div");
  empty.className = "gpv-empty";
  empty.textContent = "Queue a render to watch it denoise";
  const img = document.createElement("img");
  img.style.display = "none";
  /* an animated webp cannot be paused, so a freeze is a snapshot of the
   * currently shown frame drawn onto a canvas in its place */
  const frozen = document.createElement("canvas");
  frozen.style.display = "none";
  stage.append(empty, img, frozen);

  const meta = document.createElement("div");
  meta.className = "gpv-meta";
  const left = document.createElement("span");
  const right = document.createElement("span");
  right.className = "sp";
  const btn = document.createElement("button");
  btn.className = "gpv-btn";
  btn.textContent = "\u23F8 pause";
  btn.style.display = "none";
  meta.append(left, right, btn);

  root.append(stage, meta);

  let playing = true;

  function freeze() {
    if (!img.naturalWidth) return;
    frozen.width = img.naturalWidth;
    frozen.height = img.naturalHeight;
    frozen.getContext("2d").drawImage(img, 0, 0);
    img.style.display = "none";
    frozen.style.display = "block";
    playing = false;
    btn.textContent = "\u25B6 play";
  }

  function resume() {
    const src = img.src;
    frozen.style.display = "none";
    img.style.display = "block";
    img.src = src;                 // restart the loop from the beginning
    playing = true;
    btn.textContent = "\u23F8 pause";
  }

  btn.onclick = (e) => { e.stopPropagation(); playing ? freeze() : resume(); };
  btn.addEventListener("pointerdown", (e) => e.stopPropagation());

  return {
    root,
    freeze,
    show(d) {
      frozen.style.display = "none";
      img.src = "data:image/webp;base64," + d.webp;
      img.style.display = "block";
      empty.style.display = "none";
      btn.style.display = "";
      playing = true;
      btn.textContent = "\u23F8 pause";
      left.innerHTML = `<b>${d.fps ?? "?"} fps</b> \u00b7 ${d.frames} of ${d.latent_frames} latent frames`;
      right.textContent = `${d.seconds}s shot`;
      node.setDirtyCanvas(true, true);
    },
  };
}

app.registerExtension({
  name: "cglide.glidepreview",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);
      const ui = build(this);
      this._gpv = ui;
      this.addDOMWidget("gpv_ui", "div", ui.root, { serialize: false, hideOnZoom: false });
      nodes.add(this);
      if (this.size[0] < 320) this.size[0] = 320;
      if (this.size[1] < 380) this.size[1] = 380;
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      nodes.delete(this);
      return onRemoved?.apply(this, arguments);
    };
  },
});

api.addEventListener("csglide.preview", (e) => {
  const d = e.detail;
  if (!d || !d.webp) return;
  for (const n of nodes) n._gpv?.show(d);
});

/* stop looping once the queue is done - it plays while it is useful, then
 * holds the last frame instead of spinning forever */
const stopAll = () => { for (const n of nodes) n._gpv?.freeze(); };
api.addEventListener("execution_success", stopAll);
api.addEventListener("execution_error", stopAll);
api.addEventListener("execution_interrupted", stopAll);
api.addEventListener("executing", (e) => { if (!e.detail || e.detail.node == null) stopAll(); });
