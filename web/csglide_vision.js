import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// ---------------------------------------------------------------------------
// shared dark-grey theme (Director-style) + red hover on Analyze
// ---------------------------------------------------------------------------
function injectStyle() {
  if (document.getElementById("csglide-vision-style")) return;
  const s = document.createElement("style");
  s.id = "csglide-vision-style";
  s.textContent = [
    ".csg-btn{background:#2a2a2a;border:1px solid #424242;color:#cfcfcf;border-radius:6px;",
    "cursor:pointer;font-size:12px;transition:background .15s,border-color .15s,color .15s;}",
    ".csg-btn:hover{background:#333;border-color:#555;}",
    ".csg-gear{background:#2a2a2a;border:1px solid #424242;color:#cfcfcf;border-radius:6px;",
    "cursor:pointer;line-height:1;transition:background .15s,border-color .15s;}",
    ".csg-gear:hover{background:#333;border-color:#555;}",
    ".csg-analyze{background:#2a2a2a;border:1px solid #454545;color:#eaeaea;border-radius:7px;",
    "cursor:pointer;font-weight:600;transition:background .18s,border-color .18s,color .18s;}",
    ".csg-analyze:hover{background:linear-gradient(90deg,#333,#b0382c);border-color:#b0382c;color:#fff;}",
    ".csg-analyze:disabled{opacity:.6;cursor:default;}",
    ".csg-analyze:disabled:hover{background:#2a2a2a;border-color:#454545;color:#eaeaea;}",
    ".csg-copy{background:#2a2a2a;border:1px solid #424242;color:#cfcfcf;border-radius:6px;",
    "cursor:pointer;font-size:11px;padding:3px 10px;transition:background .15s,border-color .15s;}",
    ".csg-copy:hover{background:#333;border-color:#555;}",
  ].join("");
  document.head.appendChild(s);
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function getWidget(node, name) {
  return node.widgets ? node.widgets.find((w) => w.name === name) : null;
}
function setWidget(node, name, value) {
  const w = getWidget(node, name);
  if (w) { w.value = value; if (w.callback) w.callback(value); }
}
function hideWidget(w) {
  if (!w || w._csHidden) return;
  w._csHidden = true;
  w.hidden = true;
  w._origCompute = w.computeSize;
  w.computeSize = () => [0, -4];
  if (w.element) { w.element.style.display = "none"; w.element.hidden = true; }
}
function fileToB64(file) {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result).split(",")[1]);
    r.onerror = rej;
    r.readAsDataURL(file);
  });
}
function viewURL(name) {
  let sub = "", fn = name;
  const ix = name.lastIndexOf("/");
  if (ix >= 0) { sub = name.slice(0, ix); fn = name.slice(ix + 1); }
  const path =
    "/view?filename=" + encodeURIComponent(fn) +
    "&type=input&subfolder=" + encodeURIComponent(sub) + "&t=" + Date.now();
  return api.apiURL ? api.apiURL(path) : path;
}

const VISION_HINTS = ["vl", "vision", "llava", "moondream", "minicpm", "qwen", "gemma", "pixtral"];

// ---------------------------------------------------------------------------
function buildUI(node) {
  injectStyle();

  const wBackend = getWidget(node, "backend");
  const wUrl = getWidget(node, "server_url");
  const wModel = getWidget(node, "model");
  const wImageName = getWidget(node, "image_name");
  const wCustom = getWidget(node, "custom_prompt");
  const wDesc = getWidget(node, "description");
  const wAutoEject = getWidget(node, "auto_eject");

  [wBackend, wUrl, wModel, wImageName, wCustom, wDesc, wAutoEject].forEach(hideWidget);

  let settingsOpen = false;
  let currentB64 = null;

  const inputCss =
    "background:#1f1f1f;border:1px solid #3a3a3a;color:#e2e2e2;border-radius:6px;" +
    "padding:6px 8px;font-size:12px;outline:none;width:100%;box-sizing:border-box;";

  const root = document.createElement("div");
  root.style.cssText =
    "position:relative;display:flex;flex-direction:column;gap:7px;width:100%;box-sizing:border-box;" +
    "font-family:inherit;font-size:12px;color:#d2d2d2;padding:2px;";

  // ---- top bar -------------------------------------------------------------
  const topbar = document.createElement("div");
  topbar.style.cssText = "display:flex;align-items:center;justify-content:space-between;";
  const ttl = document.createElement("div");
  ttl.textContent = "Image describer";
  ttl.style.cssText = "opacity:.65;letter-spacing:.3px;";
  const gear = document.createElement("button");
  gear.className = "csg-gear";
  gear.textContent = "\u2699";
  gear.title = "Backend settings";
  gear.style.cssText += "width:26px;height:24px;font-size:14px;";
  topbar.appendChild(ttl);
  topbar.appendChild(gear);

  // ---- settings popover ----------------------------------------------------
  const panel = document.createElement("div");
  panel.style.cssText =
    "display:none;position:absolute;top:34px;left:2px;right:2px;z-index:30;" +
    "flex-direction:column;gap:7px;padding:9px;border:1px solid #3a3a3a;" +
    "border-radius:8px;background:#262626;box-shadow:0 8px 24px rgba(0,0,0,.6);";

  function field(label) {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-direction:column;gap:3px;";
    const l = document.createElement("label");
    l.textContent = label;
    l.style.cssText = "opacity:.6;font-size:11px;";
    row.appendChild(l);
    return row;
  }

  const rBackend = field("Backend");
  const selBackend = document.createElement("select");
  selBackend.style.cssText = inputCss;
  [["ollama", "Ollama (local)"], ["openai", "OpenAI-compatible"]].forEach(([v, t]) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = t;
    selBackend.appendChild(o);
  });
  rBackend.appendChild(selBackend);

  const rUrl = field("Server URL");
  const inUrl = document.createElement("input");
  inUrl.type = "text"; inUrl.style.cssText = inputCss;
  rUrl.appendChild(inUrl);

  const rModel = field("Model");
  const modelWrap = document.createElement("div");
  modelWrap.style.cssText = "display:flex;gap:6px;";
  const inModel = document.createElement("input");
  inModel.type = "text";
  inModel.setAttribute("list", "csmodels_" + node.id);
  inModel.placeholder = "pick or type a model";
  inModel.style.cssText = inputCss;
  const dataList = document.createElement("datalist");
  dataList.id = "csmodels_" + node.id;
  const btnRefresh = document.createElement("button");
  btnRefresh.className = "csg-btn";
  btnRefresh.textContent = "\u21bb";
  btnRefresh.title = "Refresh installed models";
  btnRefresh.style.cssText += "flex:0 0 34px;padding:6px;";
  modelWrap.appendChild(inModel);
  modelWrap.appendChild(btnRefresh);
  modelWrap.appendChild(dataList);
  rModel.appendChild(modelWrap);

  const btnEject = document.createElement("button");
  btnEject.className = "csg-btn";
  btnEject.textContent = "Eject model (free VRAM)";
  btnEject.style.cssText += "padding:7px;width:100%;margin-top:2px;";

  const autoRow = document.createElement("label");
  autoRow.style.cssText =
    "display:flex;align-items:center;gap:7px;font-size:11px;opacity:.85;cursor:pointer;margin-top:2px;";
  const autoChk = document.createElement("input");
  autoChk.type = "checkbox";
  autoChk.style.cssText = "width:14px;height:14px;cursor:pointer;accent-color:#b0382c;";
  const autoTxt = document.createElement("span");
  autoTxt.textContent = "Auto-eject model on workflow run";
  autoRow.appendChild(autoChk);
  autoRow.appendChild(autoTxt);
  autoChk.onchange = () => setWidget(node, "auto_eject", autoChk.checked);

  panel.appendChild(rBackend);
  panel.appendChild(rUrl);
  panel.appendChild(rModel);
  panel.appendChild(btnEject);
  panel.appendChild(autoRow);

  async function refreshModels() {
    if (selBackend.value !== "ollama") return;
    btnRefresh.textContent = "\u22ef";
    try {
      const res = await api.fetchApi(
        "/csglide_vision/models?url=" + encodeURIComponent(inUrl.value || "")
      );
      const j = await res.json();
      dataList.innerHTML = "";
      (j.models || []).forEach((m) => {
        const o = document.createElement("option");
        o.value = m;
        dataList.appendChild(o);
      });
      const have = j.models || [];
      if (have.length && !have.includes(inModel.value)) {
        const pick =
          have.find((m) => VISION_HINTS.some((h) => m.toLowerCase().includes(h))) || have[0];
        inModel.value = pick;
        setWidget(node, "model", pick);
      }
    } catch (e) { /* ollama probably off */ }
    btnRefresh.textContent = "\u21bb";
  }

  selBackend.onchange = () => { setWidget(node, "backend", selBackend.value); refreshModels(); };
  inUrl.oninput = () => setWidget(node, "server_url", inUrl.value);
  inModel.oninput = () => setWidget(node, "model", inModel.value);
  btnRefresh.onclick = refreshModels;

  gear.onclick = (e) => {
    e.stopPropagation();
    settingsOpen = !settingsOpen;
    panel.style.display = settingsOpen ? "flex" : "none";
    if (settingsOpen) { syncFromWidgets(); refreshModels(); }
    node.setDirtyCanvas(true, true);
  };
  const onDocPointer = (e) => {
    if (!settingsOpen) return;
    if (panel.contains(e.target) || gear.contains(e.target)) return;
    settingsOpen = false;
    panel.style.display = "none";
    node.setDirtyCanvas(true, true);
  };
  document.addEventListener("pointerdown", onDocPointer, true);
  const prevOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    document.removeEventListener("pointerdown", onDocPointer, true);
    if (prevOnRemoved) prevOnRemoved.apply(this, arguments);
  };

  btnEject.onclick = async () => {
    btnEject.disabled = true;
    const old = btnEject.textContent;
    btnEject.textContent = "Ejecting...";
    try {
      const res = await api.fetchApi("/csglide_vision/eject", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          backend: wBackend ? wBackend.value : "ollama",
          url: wUrl ? wUrl.value : "http://localhost:11434",
          model: wModel ? wModel.value : "",
        }),
      });
      const j = await res.json();
      status.textContent = j.ok ? (j.msg || "Ejected.") : "Eject error: " + j.error;
    } catch (e) {
      status.textContent = "Eject error: " + e;
    } finally {
      btnEject.disabled = false;
      btnEject.textContent = old;
    }
  };

  // ---- drop zone -----------------------------------------------------------
  const drop = document.createElement("div");
  drop.style.cssText =
    "position:relative;border:1.5px dashed #3a3a3a;border-radius:8px;min-height:120px;" +
    "display:flex;align-items:center;justify-content:center;text-align:center;" +
    "background:#1f1f1f;cursor:pointer;overflow:hidden;";
  const hint = document.createElement("div");
  hint.innerHTML = "Drop an image here<br><span style='opacity:.5'>or use the buttons below</span>";
  hint.style.cssText = "opacity:.6;padding:14px;pointer-events:none;";
  const preview = document.createElement("img");
  preview.style.cssText = "display:none;max-width:100%;max-height:220px;border-radius:6px;object-fit:contain;";
  drop.appendChild(hint);
  drop.appendChild(preview);

  function showPreviewSrc(src) {
    preview.src = src;
    preview.style.display = "block";
    hint.style.display = "none";
  }

  // ---- input buttons -------------------------------------------------------
  const btnRow = document.createElement("div");
  btnRow.style.cssText = "display:flex;gap:6px;";
  const btnBrowse = document.createElement("button");
  btnBrowse.className = "csg-btn"; btnBrowse.textContent = "Browse";
  btnBrowse.style.cssText += "flex:1;padding:7px;";
  const fileInput = document.createElement("input");
  fileInput.type = "file"; fileInput.accept = "image/*"; fileInput.style.display = "none";
  const btnUrl = document.createElement("button");
  btnUrl.className = "csg-btn"; btnUrl.textContent = "URL";
  btnUrl.style.cssText += "flex:1;padding:7px;";
  btnRow.appendChild(btnBrowse);
  btnRow.appendChild(btnUrl);

  const urlBar = document.createElement("div");
  urlBar.style.cssText = "display:none;gap:6px;";
  const urlInput = document.createElement("input");
  urlInput.type = "text"; urlInput.placeholder = "https://..."; urlInput.style.cssText = inputCss + "flex:1;";
  const urlLoad = document.createElement("button");
  urlLoad.className = "csg-btn"; urlLoad.textContent = "Load";
  urlLoad.style.cssText += "flex:0 0 60px;padding:7px;";
  urlBar.appendChild(urlInput);
  urlBar.appendChild(urlLoad);

  // ---- custom prompt -------------------------------------------------------
  const cpLabel = document.createElement("label");
  cpLabel.textContent = "Custom prompt (optional \u2014 leave empty for a Full description)";
  cpLabel.style.cssText = "opacity:.6;font-size:11px;";
  const cpArea = document.createElement("textarea");
  cpArea.rows = 2;
  cpArea.placeholder = "e.g. describe the front car in detail, focus on the bodywork...";
  cpArea.style.cssText = inputCss + "resize:vertical;min-height:38px;font-family:inherit;";
  cpArea.oninput = () => setWidget(node, "custom_prompt", cpArea.value);

  // ---- analyze + magic -----------------------------------------------------
  const analyzeRow = document.createElement("div");
  analyzeRow.style.cssText = "display:flex;gap:6px;align-items:stretch;";
  const analyze = document.createElement("button");
  analyze.className = "csg-analyze";
  analyze.textContent = "Analyze";
  analyze.style.cssText += "flex:1;padding:10px;font-size:13px;";
  const btnMagic = document.createElement("button");
  btnMagic.className = "csg-analyze";
  btnMagic.textContent = "\uD83E\uDE84"; // wand
  btnMagic.title = "Paste image or image URL from clipboard, then load & analyze";
  btnMagic.style.cssText += "flex:0 0 46px;padding:10px;font-size:15px;";
  analyzeRow.appendChild(analyze);
  analyzeRow.appendChild(btnMagic);

  // ---- description + copy --------------------------------------------------
  const dRow = document.createElement("div");
  dRow.style.cssText = "display:flex;align-items:center;justify-content:space-between;";
  const dLabel = document.createElement("label");
  dLabel.textContent = "Description (output)";
  dLabel.style.cssText = "opacity:.6;font-size:11px;";
  const btnCopy = document.createElement("button");
  btnCopy.className = "csg-copy";
  btnCopy.textContent = "Copy";
  dRow.appendChild(dLabel);
  dRow.appendChild(btnCopy);

  const dArea = document.createElement("textarea");
  dArea.rows = 4;
  dArea.style.cssText = inputCss + "resize:vertical;min-height:74px;font-family:inherit;";
  dArea.oninput = () => setWidget(node, "description", dArea.value);

  btnCopy.onclick = async () => {
    const txt = dArea.value || "";
    if (!txt) return;
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(txt);
      } else {
        dArea.select(); document.execCommand("copy");
        window.getSelection().removeAllRanges();
      }
      btnCopy.textContent = "Copied!";
    } catch (e) {
      btnCopy.textContent = "Failed";
    }
    setTimeout(() => (btnCopy.textContent = "Copy"), 1200);
  };

  const status = document.createElement("div");
  status.style.cssText = "min-height:14px;font-size:11px;opacity:.75;";

  // assemble
  root.appendChild(topbar);
  root.appendChild(panel);
  root.appendChild(drop);
  root.appendChild(btnRow);
  root.appendChild(urlBar);
  root.appendChild(cpLabel);
  root.appendChild(cpArea);
  root.appendChild(analyzeRow);
  root.appendChild(dRow);
  root.appendChild(dArea);
  root.appendChild(status);
  root.appendChild(fileInput);

  // ---- sync UI from saved widget values (workflow load / tab switch) -------
  function syncFromWidgets() {
    if (wBackend) selBackend.value = wBackend.value || "ollama";
    if (wUrl) inUrl.value = wUrl.value || "http://localhost:11434";
    if (wModel) inModel.value = wModel.value || "";
    if (wCustom) cpArea.value = wCustom.value || "";
    if (wDesc) dArea.value = wDesc.value || "";
    if (wAutoEject) autoChk.checked = !!wAutoEject.value;
    const nm = wImageName ? (wImageName.value || "") : "";
    if (nm) { showPreviewSrc(viewURL(nm)); }
    else { preview.style.display = "none"; hint.style.display = "block"; }
  }

  // ---- image handling ------------------------------------------------------
  async function handleFile(file) {
    if (!file) return;
    try {
      status.textContent = "Loading image...";
      const b64 = await fileToB64(file);
      currentB64 = b64;
      showPreviewSrc("data:" + (file.type || "image/png") + ";base64," + b64);
      const fd = new FormData();
      fd.append("image", file, file.name || "csglide_input.png");
      fd.append("overwrite", "true");
      const res = await api.fetchApi("/upload/image", { method: "POST", body: fd });
      const j = await res.json();
      const nm = j.subfolder ? j.subfolder + "/" + j.name : j.name;
      setWidget(node, "image_name", nm);
      status.textContent = "Ready.";
    } catch (e) {
      status.textContent = "Upload error: " + e;
    }
  }

  fileInput.onchange = () => handleFile(fileInput.files[0]);
  btnBrowse.onclick = () => fileInput.click();
  drop.onclick = () => fileInput.click();
  drop.ondragover = (e) => { e.preventDefault(); drop.style.borderColor = "#888"; };
  drop.ondragleave = () => { drop.style.borderColor = "#3a3a3a"; };
  drop.ondrop = (e) => {
    e.preventDefault();
    drop.style.borderColor = "#3a3a3a";
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  };

  btnUrl.onclick = () => {
    urlBar.style.display = urlBar.style.display === "flex" ? "none" : "flex";
  };
  async function loadFromUrl(u) {
    if (!u) return;
    status.textContent = "Fetching URL...";
    const res = await api.fetchApi("/csglide_vision/load_url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: u }),
    });
    const j = await res.json();
    if (!j.ok) throw new Error(j.error || "failed");
    currentB64 = j.image_b64;
    showPreviewSrc("data:image/png;base64," + j.image_b64);
    setWidget(node, "image_name", j.name);
    status.textContent = "Ready.";
  }
  urlLoad.onclick = () => {
    loadFromUrl(urlInput.value.trim()).catch((e) => (status.textContent = "URL error: " + e));
  };

  // ---- analyze -------------------------------------------------------------
  async function runAnalyze() {
    const haveName = wImageName && wImageName.value;
    if (!currentB64 && !haveName) { status.textContent = "Load an image first."; return; }
    analyze.disabled = true;
    btnMagic.disabled = true;
    analyze.textContent = "Analyzing...";
    status.textContent = "Talking to " + (wBackend ? wBackend.value : "ollama") + "...";
    try {
      const res = await api.fetchApi("/csglide_vision/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: currentB64 || "",
          name: wImageName ? wImageName.value : "",
          prompt: wCustom ? wCustom.value : "",
          backend: wBackend ? wBackend.value : "ollama",
          url: wUrl ? wUrl.value : "http://localhost:11434",
          model: wModel ? wModel.value : "llava",
        }),
      });
      const j = await res.json();
      if (!j.ok) throw new Error(j.error || "failed");
      dArea.value = j.description;
      setWidget(node, "description", j.description);
      status.textContent = "Done.";
      node.setDirtyCanvas(true, true);
    } catch (e) {
      status.textContent = "Error: " + e;
    } finally {
      analyze.disabled = false;
      btnMagic.disabled = false;
      analyze.textContent = "Analyze";
    }
  }
  analyze.onclick = runAnalyze;

  // ---- magic: paste from clipboard (image or URL), load, analyze ----------
  async function readClipboardImage() {
    if (!navigator.clipboard || !navigator.clipboard.read) return null;
    const items = await navigator.clipboard.read();
    for (const it of items) {
      const t = it.types.find((x) => x.startsWith("image/"));
      if (t) {
        const blob = await it.getType(t);
        return new File([blob], "clipboard.png", { type: blob.type });
      }
    }
    return null;
  }

  btnMagic.onclick = async () => {
    btnMagic.disabled = true;
    const old = btnMagic.textContent;
    btnMagic.textContent = "\u2026";
    status.textContent = "Reading clipboard...";
    try {
      let loaded = false;
      try {
        const f = await readClipboardImage();
        if (f) { await handleFile(f); loaded = true; }
      } catch (e) { /* no image in clipboard / not permitted */ }

      if (!loaded) {
        let txt = "";
        try { txt = ((await navigator.clipboard.readText()) || "").trim(); } catch (e) {}
        if (/^https?:\/\//i.test(txt)) { await loadFromUrl(txt); loaded = true; }
      }

      if (!loaded && !currentB64 && !(wImageName && wImageName.value)) {
        status.textContent = "Clipboard has no image or image URL.";
        return;
      }
      btnMagic.textContent = old;
      await runAnalyze();
    } catch (e) {
      status.textContent = "Magic error: " + e;
    } finally {
      btnMagic.textContent = old;
      btnMagic.disabled = false;
    }
  };

  // ---- mount + persistence hooks ------------------------------------------
  node.addDOMWidget("csglide_ui", "div", root, { serialize: false });
  syncFromWidgets();

  const prevConfigure = node.onConfigure;
  node.onConfigure = function () {
    const r = prevConfigure ? prevConfigure.apply(this, arguments) : undefined;
    setTimeout(syncFromWidgets, 0);
    return r;
  };

  node.size = [340, 620];
}

app.registerExtension({
  name: "CSGlide.Vision",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "CSGlideVisionCS") return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      buildUI(this);
      return r;
    };
  },
});
