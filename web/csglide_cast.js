import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/* =========================================================================
 * H3 Studio — UI
 *
 * Two model families, one node. The accent colour IS the family indicator:
 *   blue  = fl2va  (first / last keyframes)
 *   green = ref2va (9 images, 3 videos, 3 audio)
 * Mismatched weights run fine and produce nonsense, so the node shouts which
 * family it is currently asking for.
 * ====================================================================== */

const NODE_ID = "CSGlideCastCS";
const ASSET_SUBFOLDER = "cglide";
const FPS = 24;
const MAX_IMAGES = 9, MAX_VIDEOS = 3, MAX_AUDIOS = 3;

/* Ratio families. The first size in each ladder is the canvas H3's own
 * adapt_canvas() would pick; everything below it holds the same aspect on a
 * shorter edge, every axis a multiple of 32. */
const MAX_PIXELS = 768 * 1344;

const RATIOS = [
  { label: "21:9", w: 1536, h: 672 },
  { label: "16:9", w: 1344, h: 768 },
  { label: "3:2",  w: 1152, h: 768 },
  { label: "4:3",  w: 1024, h: 768 },
  { label: "1:1",  w: 768,  h: 768 },
  { label: "3:4",  w: 768,  h: 1024 },
  { label: "2:3",  w: 768,  h: 1152 },
  { label: "9:16", w: 768,  h: 1344 },
];

const SHORT_EDGES = [768, 704, 640, 576, 512, 448, 384, 352, 320];

function sizeLadder(base) {
  const r = base.w / base.h;
  const seen = new Set(), out = [];
  for (const shortEdge of SHORT_EDGES) {
    let w, h;
    if (r >= 1) { h = shortEdge; w = Math.max(32, Math.round((shortEdge * r) / 32) * 32); }
    else { w = shortEdge; h = Math.max(32, Math.round((shortEdge / r) / 32) * 32); }
    if (w * h > MAX_PIXELS + 1) continue;
    const key = `${w}x${h}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push({ w, h });
  }
  if (!out.some((o) => o.w === base.w && o.h === base.h)) out.unshift({ w: base.w, h: base.h });
  return out;
}

RATIOS.forEach((r) => { r.sizes = sizeLadder(r); });

function findRatio(w, h) {
  for (const r of RATIOS) if (r.sizes.some((s) => s.w === w && s.h === h)) return r;
  return null;
}

/* Frame counts on the 17k+5 grid. Seconds shown are the real duration. */
const LENGTH_PRESETS = [56, 73, 107, 124, 158, 192, 226, 243, 294, 328, 362];

const ALL_TOKENS = [
  "@first", "@last",
  ...Array.from({ length: MAX_IMAGES }, (_, i) => `@image${i + 1}`),
  ...Array.from({ length: MAX_VIDEOS }, (_, i) => `@video${i + 1}`),
  ...Array.from({ length: MAX_VIDEOS }, (_, i) => `@videoaudio${i + 1}`),
  ...Array.from({ length: MAX_AUDIOS }, (_, i) => `@audio${i + 1}`),
];

const MIN_H_REF = 1252, MIN_H_FL = 1252;   // same floor in both modes, so the panel does not shift
/* v2 sizes the DOM widget from its content, so a shorter mode leaves a gap
 * above the panel. Pinning one content height keeps the top edge put. */
const CONTENT_MIN = 1140;
const PROMPT_H_REF = 160, PROMPT_H_FL = 300;   // fl2va has only two slots, so the space goes to the prompt

const alignFrames = (n) => { n = Math.max(5, Math.round(n)); while (n % 17 !== 5) n++; return n; };
const fmtSecs = (s) => (Math.round(s * 100) / 100).toFixed(2) + "s";

/* ---------------------------------------------------------------- state */

function blankState() {
  const mk = (n) => Array.from({ length: n }, () => ({}));
  return {
    mode: "ref2va",
    width: 1344, height: 768,
    length: 243,
    ref_image_size: "match",
    pace: 2.5,          /* legacy: the speech-budget slider is gone, kept so old .h3.json still parses */
    prompt: "",
    slots: { first: {}, last: {}, images: mk(MAX_IMAGES), videos: mk(MAX_VIDEOS), audios: mk(MAX_AUDIOS) },
    /* CONTINUE FROM: the previous clip's tail, anchored at frame 0 of this one.
     * Deliberately NOT a reference slot - it takes no @token, never appears in
     * the presentation, and works in both modes. Continuing is orthogonal to
     * first/last vs omni, so it is a section rather than a third mode. */
    cont: {},
  };
}

/* Whitelisted parse. Anything not named here is dropped on reload — on purpose.
 * Adding a new stored field means adding it HERE too, or it vanishes silently. */
function parseInitial(raw) {
  const out = blankState();
  let d = {};
  try { d = typeof raw === "string" ? JSON.parse(raw || "{}") : (raw || {}); } catch (e) { d = {}; }
  if (!d || typeof d !== "object") d = {};

  if (d.mode === "fl2va") out.mode = "fl2va";
  if (Number.isFinite(+d.width)) out.width = +d.width;
  if (Number.isFinite(+d.height)) out.height = +d.height;
  if (Number.isFinite(+d.length)) out.length = alignFrames(+d.length);
  if (d.ref_image_size === "max") out.ref_image_size = "max";
  if (Number.isFinite(+d.pace) && +d.pace > 0) out.pace = Math.min(4, Math.max(1, +d.pace));
  if (typeof d.prompt === "string") out.prompt = d.prompt;

  const s = (d.slots && typeof d.slots === "object") ? d.slots : {};
  const image = (o) => (o && o.file) ? { file: String(o.file) } : {};
  const timed = (o, withAudio) => {
    if (!o || !o.file) return {};
    const r = { file: String(o.file) };
    if (Number.isFinite(+o.start)) r.start = +o.start;
    if (Number.isFinite(+o.end)) r.end = +o.end;
    if (Number.isFinite(+o.dur)) r.dur = +o.dur;
    if (withAudio && o.audio) r.audio = true;
    return r;
  };

  out.slots.first = image(s.first);
  out.slots.last = image(s.last);
  for (let i = 0; i < MAX_IMAGES; i++) out.slots.images[i] = image((s.images || [])[i]);
  for (let i = 0; i < MAX_VIDEOS; i++) out.slots.videos[i] = timed((s.videos || [])[i], true);
  for (let i = 0; i < MAX_AUDIOS; i++) out.slots.audios[i] = timed((s.audios || [])[i], false);

  const c = (d.cont && typeof d.cont === "object") ? d.cont : {};
  if (c.file) {
    out.cont = { file: String(c.file) };
    if (Number.isFinite(+c.start)) out.cont.start = +c.start;
    if (Number.isFinite(+c.end)) out.cont.end = +c.end;
    if (Number.isFinite(+c.dur)) out.cont.dur = +c.dur;
    if (c.audio) out.cont.audio = true;
    if (c.flatten) out.cont.flatten = true;
  }
  return out;
}

/* Same ordinal rules as the tokenizer: images, then videos (a soundtrack's
 * <Audio j> lands immediately before its <Video k>), then standalone audio.
 * Ordinals are 1-based per type and count only filled slots. */
function presentation(st) {
  const rows = [], tags = {};
  if (st.mode === "fl2va") {
    let i = 0;
    if (st.slots.first.file) { i++; tags["@first"] = `<Picture ${i}>`; rows.push({ tag: `<Picture ${i}>`, from: "first frame", kind: "image" }); }
    if (st.slots.last.file)  { i++; tags["@last"]  = `<Picture ${i}>`; rows.push({ tag: `<Picture ${i}>`, from: "last frame",  kind: "image" }); }
    return { rows, tags };
  }
  let i = 0, j = 0, k = 0;
  st.slots.images.forEach((s, n) => {
    if (!s.file) return; i++;
    tags[`@image${n + 1}`] = `<Picture ${i}>`;
    rows.push({ tag: `<Picture ${i}>`, from: `image ${n + 1}`, kind: "image", token: `@image${n + 1}` });
  });
  st.slots.videos.forEach((s, n) => {
    if (!s.file) return;
    if (s.audio) {
      j++; tags[`@videoaudio${n + 1}`] = `<Audio ${j}>`;
      rows.push({ tag: `<Audio ${j}>`, from: `video ${n + 1} sound`, kind: "audio", token: `@videoaudio${n + 1}` });
    }
    k++; tags[`@video${n + 1}`] = `<Video ${k}>`;
    rows.push({ tag: `<Video ${k}>`, from: `video ${n + 1}`, kind: "video", token: `@video${n + 1}` });
  });
  st.slots.audios.forEach((s, n) => {
    if (!s.file) return; j++;
    tags[`@audio${n + 1}`] = `<Audio ${j}>`;
    rows.push({ tag: `<Audio ${j}>`, from: `audio ${n + 1}`, kind: "audio", token: `@audio${n + 1}` });
  });
  return { rows, tags };
}

/* ---- prompt check ---------------------------------------------------
 * H3 always generates a soundtrack for the full duration, so whatever the
 * prompt leaves unsaid gets invented -- which is where mumbling comes from.
 * Counting words against a pace was the wrong measure: a well-paced line
 * still leaves seconds of unassigned audio at the end of the shot. What
 * actually stopped it in testing was structural, so this looks for the
 * structure: named fields, a speaker ID, mouths closed after the line, and
 * a soundscape that covers the whole clip.
 */

/* average delivery rate. Only an estimate of how long the written lines take
 * to say -- it is NOT a target to fill, which is what the old budget got
 * wrong. Coverage of the tail is handled by the soundscape row. */
const WORDS_PER_SEC = 2.5;

const FIELD_BODY  = /^[ \t]*(?:detailed_description|integrated_multimodal_description)[ \t]*:/im;
const FIELD_SOUND = /^[ \t]*overall_soundscape[ \t]*:/im;
const FIELD_MUSIC = /^[ \t]*non_diegetic_music[ \t]*:/im;

const DTAG_RE = /<d>\s*(?:\[[^\]]*\]\s*)?([^<]{1,400})<\/d>/gi;
const QUOTE_RE = /["\u201c]([^"\u201d]{1,400})["\u201d]/g;
/* a quote introduced like this is on-screen text, not speech */
const SCREEN_TEXT = /(read(?:s|ing)?|sign|banner|label|subtitle|caption|placard|logo|title card|neon|lettering|letters|text|marked|stencilled|stenciled|written)\W*$/i;
/* the official form is (S1); (S1,S2) covers simultaneous speech */
const SPEAKER_RE = /\(\s*S\d+\s*(?:,\s*S\d+\s*)*\)/;
/* mouths shut, stated positively -- H3 is CFG-distilled, so "no murmuring"
 * has nothing to push against and lands weakly */
const LIPS_RE = /lips?\s+(?:close|remain|stay|are\s+close)|close[sd]?\s+(?:his|her|their|the)\s+(?:lips|mouth)|mouths?\s+(?:closed|stay|remain)|(?:does|do)\s+not\s+speak\s+again|never\s+speaks?|says?\s+nothing\s+(?:more|further)|no\s+further\s+dialogue/i;
const SILENT_INTENT = /silen|wordless|mute|no speech|no dialogue|no words|no line|not speak|doesn't speak|says? nothing|saying nothing|without speaking|(?:neither|nobody|no one|no-one) speaks?|stays? quiet|remains? quiet|in silence|beat of quiet/i;

/* text of a named field, up to the next field header or the end of the prompt */
const ANY_FIELD = /^[ \t]*[a-z_]{4,}[ \t]*:/gm;
function fieldText(prompt, name) {
  const head = new RegExp("^[ \\t]*" + name + "[ \\t]*:", "im").exec(prompt);
  if (!head) return null;
  const from = head.index + head[0].length;
  let stop = prompt.length, m;
  ANY_FIELD.lastIndex = from;
  while ((m = ANY_FIELD.exec(prompt)) !== null) { stop = m.index; break; }
  return prompt.slice(from, stop).trim();
}

/* Shot markers, official format: "[Shot 2] At 00:03.500," -- Shot 1 carries no
 * timestamp and starts at zero. A shot runs until the next one starts, or to
 * the end of the clip. "At 3.5s," and "At 00:03," are tolerated too, since the
 * guide's MM:SS.mmm is not what people type by hand.
 *
 * The END of a shot is never stated, so it is always inferred. That is the
 * format's own rule, not a guess: shots are contiguous inside one generation. */
const SHOT_RE = /\[\s*shot\s*(\d+)\s*\]([^\n]{0,90})/gi;
const AT_MMSS = /\bat\s+(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?/i;
const AT_SECS = /\bat\s+(\d{1,3}(?:[.,]\d+)?)\s*s\b/i;

function parseShots(prompt, total) {
  const p = prompt || "";
  const marks = [];
  let m;
  SHOT_RE.lastIndex = 0;
  while ((m = SHOT_RE.exec(p)) !== null) {
    const tail = m[2] || "";
    const tailAt = m.index + (m[0].length - tail.length);
    const a = AT_MMSS.exec(tail), b = a ? null : AT_SECS.exec(tail);
    let at = null, tsFrom = null, tsTo = null;
    if (a) {
      at = (+a[1]) * 60 + (+a[2]) + (a[3] ? +("0." + a[3].padEnd(3, "0")) : 0);
      tsFrom = tailAt + a.index; tsTo = tsFrom + a[0].length;
    } else if (b) {
      at = +String(b[1]).replace(",", ".");
      tsFrom = tailAt + b.index; tsTo = tsFrom + b[0].length;
    }
    marks.push({ n: +m[1], at, tsFrom, tsTo, afterBracket: tailAt, idx: m.index });
  }
  if (!marks.length) return [];
  if (marks[0].at == null) marks[0].at = 0;

  return marks.map((s, i) => {
    const next = marks[i + 1];
    const end = (next && next.at != null) ? next.at : total;
    return {
      n: s.n,
      start: s.at,
      end: s.at == null ? null : Math.max(s.at, end),
      untimed: s.at == null,
      past: s.at != null && s.at >= total,
      /* character ranges, so the timeline can edit the prompt back */
      tsFrom: s.tsFrom, tsTo: s.tsTo, afterBracket: s.afterBracket,
      idx: s.idx, bodyFrom: s.tsTo != null ? s.tsTo : s.afterBracket,
      bodyTo: next ? next.idx : null,
    };
  });
}

function spokenLines(text) {
  const out = [];
  let m;
  DTAG_RE.lastIndex = 0;
  while ((m = DTAG_RE.exec(text)) !== null) out.push(m[1].trim());
  if (out.length) return out;          /* tagged prompt: quotes are screen text */
  QUOTE_RE.lastIndex = 0;
  while ((m = QUOTE_RE.exec(text)) !== null) {
    const before = text.slice(Math.max(0, m.index - 48), m.index);
    if (SCREEN_TEXT.test(before)) continue;
    out.push(m[1].trim());
  }
  return out;
}

function promptCheck(prompt, totalSeconds) {
  const p = prompt || "";
  const clip = Math.max(0, +totalSeconds || 0);
  const rows = [];
  if (!p.trim()) return rows;

  /* 1. named fields */
  const missing = [];
  if (!FIELD_SOUND.test(p)) missing.push("overall_soundscape");
  if (!FIELD_MUSIC.test(p)) missing.push("non_diegetic_music");
  if (!FIELD_BODY.test(p) && missing.length === 2) {
    rows.push({ label: "format", text: "free prose \u2014 use the named fields", state: "warn" });
  } else if (missing.length) {
    rows.push({ label: "format", text: "no " + missing.join(", "), state: "warn" });
  } else {
    rows.push({ label: "format", text: "fields ok", state: "ok" });
  }

  /* 2. dialogue and speaker id */
  const body = fieldText(p, "detailed_description")
            || fieldText(p, "integrated_multimodal_description")
            || p;
  const lines = spokenLines(body);
  if (!lines.length) {
    rows.push({ label: "dialogue", text: SILENT_INTENT.test(body) ? "silent by design" : "none", state: "ok" });
  } else if (!SPEAKER_RE.test(body)) {
    rows.push({ label: "dialogue", text: lines.length + " line \u00b7 no (S1)", state: "warn" });
  } else {
    rows.push({ label: "dialogue", text: lines.length + (lines.length > 1 ? " lines" : " line"), state: "ok" });
  }

  /* 3. mouths closed after the last line */
  if (lines.length) {
    rows.push(LIPS_RE.test(body)
      ? { label: "lips", text: "closed after the line", state: "ok" }
      : { label: "lips", text: "left open \u2014 close them", state: "warn" });
  }

  /* 4. how long the written lines take to say, at an average rate */
  if (lines.length) {
    const words = lines.join(" ").split(/\s+/).filter(Boolean).length;
    const spoken = words / WORDS_PER_SEC;
    const text = `~${spoken.toFixed(1)}s of ${clip.toFixed(1)}s`;
    rows.push(spoken > clip
      ? { label: "speech", text: text + " \u00b7 too long", state: "warn" }
      : { label: "speech", text, state: "ok" });
  }

  /* 5. soundscape covering the whole clip */
  const sound = fieldText(p, "overall_soundscape");
  if (sound === null) {
    rows.push({ label: "sound", text: "unassigned", state: "warn" });
  } else {
    const w = sound.split(/\s+/).filter(Boolean).length;
    rows.push(w < 8
      ? { label: "sound", text: w + "w \u00b7 thin", state: "warn" }
      : { label: "sound", text: w + "w", state: "ok" });
  }

  return rows;
}

/* ------------------------------------------------------------------ css */

const CSS = `
.gcast {
  --h3-bg:#191919; --h3-panel:#212121; --h3-raise:#2b2b2b; --h3-line:#3b3b3b;
  --h3-well:#131313;
  --h3-txt:#e3e3e3; --h3-dim:#979797; --h3-label:#787878;
  --h3-accent:#58d1ff; --h3-accent-dim:#58d1ff26;
  --gc-wave:#59c14f;
  font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  color: var(--h3-txt); font-size: 12px; line-height: 1.35;
  position:relative;
  background: var(--h3-bg); border:1px solid var(--h3-line); border-radius:10px;
  padding:10px; display:flex; flex-direction:column; gap:9px;
  box-sizing:border-box; height:100%; overflow:auto;
}
.gcast[data-mode="fl2va"] { --h3-accent:#59c14f; --h3-accent-dim:#59c14f26; }

/* custom dropdown -------------------------------------------------- */
.gcast-select { all:unset; pointer-events:auto; cursor:pointer; box-sizing:border-box;
  display:flex; align-items:center; gap:8px; min-width:0;
  background:var(--h3-bg); color:var(--h3-txt); border:1px solid var(--h3-line);
  border-radius:5px; padding:4px 8px; font-size:11.5px; font-family:inherit; }
.gcast-select:hover { border-color:var(--h3-dim); }
.gcast-select:focus-visible { outline:2px solid var(--h3-accent); outline-offset:1px; }
.gcast-select .lbl { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gcast-select .caret { margin-left:auto; color:var(--h3-dim); font-size:9px; }
.gcast-select[aria-disabled="true"] { opacity:.45; cursor:default; }

.gcast-menu {
  --h3-panel:#212121; --h3-raise:#2b2b2b; --h3-line:#3b3b3b; --h3-well:#131313;
  --h3-txt:#e3e3e3; --h3-dim:#979797; --h3-accent:#58d1ff;
  position:fixed; z-index:9100; box-sizing:border-box;
  font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  background:var(--h3-panel); border:1px solid var(--h3-line); border-radius:11px;
  padding:7px; display:flex; flex-direction:column; gap:5px;
  max-height:60vh; overflow:auto; box-shadow:0 22px 60px #000d, 0 2px 10px #0009; }
.gcast-menu[data-mode="fl2va"] { --h3-accent:#59c14f; }
.gcast-menu button { all:unset; pointer-events:auto; cursor:pointer; box-sizing:border-box;
  display:flex; align-items:center; gap:10px; width:100%;
  background:var(--h3-raise); border:1px solid var(--h3-line); border-radius:8px;
  padding:9px 13px; font-size:12.5px; color:var(--h3-txt); white-space:nowrap; }
.gcast-menu button:hover { border-color:var(--h3-accent); }
.gcast-menu button[aria-selected="true"] { color:var(--h3-accent); border-color:var(--h3-accent); }
.gcast-menu button .note { margin-left:auto; color:var(--h3-dim); font-size:10.5px;
  font-family:ui-monospace,Consolas,monospace; }
.gcast-menu .grp { padding:3px 4px 1px; font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--h3-dim); }

.gcast-label { font-size:9.5px; letter-spacing:.13em; text-transform:uppercase;
  color:var(--h3-label); font-weight:600; }

/* mode bar ---------------------------------------------------------- */
.gcast-modebar { display:flex; align-items:center; gap:10px;
  background:linear-gradient(90deg, var(--h3-accent-dim), transparent 65%);
  border:1px solid var(--h3-line); border-left:3px solid var(--h3-accent);
  border-radius:8px; padding:7px 10px; }
.gcast-seg { display:flex; background:var(--h3-bg); border:1px solid var(--h3-line);
  border-radius:7px; padding:2px; gap:2px; }
.gcast-seg button { all:unset; pointer-events:auto; cursor:pointer; padding:5px 13px; border-radius:5px;
  font-size:11.5px; font-weight:600; color:var(--h3-dim); transition:.13s; }
.gcast-seg button:hover { color:var(--h3-txt); }
.gcast-seg button[aria-pressed="true"] { background:var(--h3-accent); color:#101010; }
.gcast-seg button:focus-visible { outline:2px solid var(--h3-accent); outline-offset:1px; }
.gcast-need { margin-left:auto; font-family:ui-monospace,Consolas,monospace; font-size:10.5px;
  color:var(--h3-dim); }
.gcast-need b { color:var(--h3-accent); font-weight:600; }

/* settings row ------------------------------------------------------ */
.gcast-row { display:grid; grid-template-columns:1.15fr 1.15fr 1fr; gap:8px; }
.gcast-card { background:var(--h3-panel); border:1px solid var(--h3-line);
  border-radius:8px; padding:8px 9px; display:flex; flex-direction:column; gap:6px;
  box-shadow:0 2px 6px rgba(0,0,0,.45); }
.gcast-ctl { display:flex; align-items:center; gap:6px; }
.gcast select, .gcast input[type="number"] {
  background:var(--h3-bg); color:var(--h3-txt); border:1px solid var(--h3-line);
  border-radius:5px; padding:4px 6px; font-size:11.5px; font-family:inherit; outline:none; }
.gcast select:focus-visible, .gcast input:focus-visible, .gcast textarea:focus-visible {
  border-color:var(--h3-accent); }
.gcast input[type="number"] { width:62px; }
.gcast-read { font-family:ui-monospace,Consolas,monospace; font-size:11px; color:var(--h3-dim); }
.gcast-read b { color:var(--h3-txt); font-weight:600; }
.gcast-toggle { display:flex; background:var(--h3-bg); border:1px solid var(--h3-line);
  border-radius:6px; padding:2px; gap:2px; }
.gcast-toggle button { all:unset; pointer-events:auto; cursor:pointer; padding:3px 9px; border-radius:4px;
  font-size:11px; color:var(--h3-dim); }
.gcast-toggle button[aria-pressed="true"] { background:var(--h3-raise); color:var(--h3-txt); }
/* max keeps references at up to 2048 short edge, which is ~4x the reference
   tokens of match riding every sampling step. Worth it for fine detail, a
   quiet way to lose a lot of VRAM otherwise -- so it reads as a warning. */
.gcast-toggle.warn button.max[aria-pressed="true"] { background:#8e3d38; color:#fff; }

/* slots ------------------------------------------------------------- */
.gcast-grid { display:grid; gap:7px; }
.gcast-grid.img { grid-template-columns:repeat(9, minmax(0,1fr)); }
.gcast-grid.fl  { grid-template-columns:repeat(2, minmax(0,1fr)); }
.gcast-grid.med { grid-template-columns:repeat(3, minmax(0,1fr)); }
/* One extra column so the continuation rides in the SAME row as the reference
   slots. It had a section of its own below them, and that cost a whole row of
   node height for one card. */
.gcast-grid.med4 { grid-template-columns:repeat(4, minmax(0,1fr)); }
.gcast-grid.fl3  { grid-template-columns:repeat(3, minmax(0,1fr)); }

.gcast-slot { position:relative; background:var(--h3-panel); border:1px solid var(--h3-line);
  border-radius:7px; overflow:hidden; cursor:pointer; transition:.13s;
  box-shadow:0 2px 6px rgba(0,0,0,.45); }
.gcast-slot:hover { border-color:var(--h3-dim); }
.gcast-slot.filled { border-color:var(--h3-accent); }
.gcast-slot.drop { border-color:var(--h3-accent); background:var(--h3-accent-dim); }
/* While a file is over the node, every slot that could take it lifts out of
   the background, so a near-miss is visible before the release rather than
   after. The slot actually under the pointer keeps the stronger .drop look. */
.gcast-armed { outline:1px dashed var(--h3-accent); outline-offset:-2px;
  background:var(--h3-accent-dim); }
.gcast-armed .gcast-empty { color:var(--h3-accent); opacity:1; }
.gcast-reject { border-color:#b4544f !important; background:rgba(180,84,79,.14) !important; }
/* Only the slot UNDER THE POINTER reads as the target. Arming every slot that
   could take the file lit up all nine images or all four video cards at once,
   which reads as everything-selected rather than as a hint. Neutralised per
   slot type, not by one blanket rule: gcast-armed lands on the THUMB for a
   video and on the card for an image, so a single background:transparent took
   the dark well with it. Each gets its own background back. Two classes beats
   the one-class gcast-armed rules above, and source order settles the rest. */
.gcast-thumb.gcast-armed { outline:none; background:var(--h3-well); }
.gcast-wav.gcast-armed   { outline:none; background:var(--h3-well); }
.gcast-slot.gcast-armed  { outline:none; background:var(--h3-panel); }
.gcast-armed .gcast-empty { color:#525252; opacity:.9; }
/* The hovered target. Media slots wire their drop on an INNER element - the
   thumb for video, the waveform for an empty audio - so the drop class lands
   there and the gcast-slot.drop rule never matched them. Without this the slot
   under the pointer showed nothing, which is why they all looked the same. */
.gcast-media.drop, .gcast-media .gcast-thumb.drop, .gcast-media .gcast-wav.drop {
  border-color:var(--h3-accent); background:var(--h3-accent-dim);
  box-shadow:inset 0 0 0 1px var(--h3-accent); }
.gcast-thumb { width:100%; aspect-ratio:1/1; display:flex; align-items:center; justify-content:center;
  background:var(--h3-well); }
.gcast-grid.fl .gcast-thumb, .gcast-grid.fl3 .gcast-thumb { aspect-ratio:16/9; }
.gcast-thumb img, .gcast-thumb video { width:100%; height:100%; object-fit:cover; display:block; }
.gcast-empty { color:#525252; font-size:18px; font-weight:300; }
.gcast-cap { display:flex; align-items:center; gap:5px; padding:4px 6px;
  font-family:ui-monospace,Consolas,monospace; font-size:10px; color:var(--h3-dim); }
.gcast-slot.filled .gcast-cap { color:var(--h3-accent); }
.gcast-cap .n { opacity:.55; }
.gcast-x { all:unset; pointer-events:auto; cursor:pointer; margin-left:auto; color:var(--h3-dim); padding:0 3px;
  border-radius:3px; font-size:12px; line-height:1; }
.gcast-x:hover { color:#ff7a7a; background:#ff7a7a1a; }

/* media cards ------------------------------------------------------- */
.gcast-media { background:var(--h3-panel); border:1px solid var(--h3-line); border-radius:7px;
  padding:7px; display:flex; flex-direction:column; gap:6px; }
.gcast-media.filled { border-color:var(--h3-accent); }
/* Amber, not the accent: the continuation is not a reference, it is where this
   clip picks up from. Different job, different colour, so the eye never files
   it with the slots either side of it. */
.gcast-media.cont { background:#221c15; border-color:#4d3b22; }
.gcast-media.cont.filled { border-color:#ff9f43; }
.gcast-media.cont .gcast-cap { color:#e8a758; }
.gcast-media.cont .gcast-ico { color:#ff9f43; display:flex; flex:0 0 auto; }
.gcast-media.cont .gcast-chk { color:#e8a758; }
.gcast-media .gcast-thumb { aspect-ratio:16/9; border-radius:5px; overflow:hidden; cursor:pointer; }
/* An extra column makes every card narrower, and a 16/9 thumb then makes the
   whole row SHORTER - spare height goes to the prompt, so the prompt visibly
   grew. Squarer thumbs put the row back at the height it had with three cards,
   and nothing below it moves.
   Source order matters here: same specificity as the 16/9 rule above, so this
   has to come AFTER it or it never applies. */
.gcast-grid.med4 .gcast-thumb { aspect-ratio:4/3; }
.gcast-grid.fl3 .gcast-thumb { aspect-ratio:6/5; }
.gcast-wav { width:100%; height:34px; background:var(--h3-well); border-radius:5px; cursor:pointer;
  display:flex; align-items:center; justify-content:center; color:#525252; font-size:11px; }

.gcast-trim { padding:2px 0 0; }
.gcast-track { position:relative; height:44px; cursor:pointer;
  background:var(--h3-well); border-radius:5px; overflow:hidden; }
.gcast-wave { position:absolute; inset:0; width:100%; height:100%; display:block; }
.gcast-wavlabel { position:absolute; inset:0; display:flex; align-items:flex-start;
  justify-content:center; padding-top:2px; font-size:10.5px; color:var(--h3-dim);
  pointer-events:none; text-shadow:0 1px 4px #000, 0 0 10px #000; letter-spacing:.02em; }
.gcast-span { position:absolute; top:0; bottom:0; background:var(--gc-wave);
  opacity:.13; pointer-events:none; }
.gcast-track.slidable { cursor:grab; }
.gcast-track.sliding { cursor:grabbing; }
.gcast-track.sliding .gcast-span { opacity:.24; }
.gcast-used { position:absolute; top:0; bottom:0; border-right:1px dashed var(--h3-dim); }
.gcast-head { position:absolute; top:0; bottom:0; width:1px; background:#fff;
  opacity:.85; display:none; pointer-events:none; }
.gcast-play { all:unset; pointer-events:auto; cursor:pointer; width:22px; height:22px; border-radius:50%;
  display:flex; align-items:center; justify-content:center; flex:0 0 auto;
  border:1px solid var(--h3-line); color:var(--h3-dim); font-size:9px; }
.gcast-play:hover { border-color:var(--h3-accent); color:var(--h3-accent); }
.gcast-play:focus-visible { outline:2px solid var(--h3-accent); outline-offset:1px; }
.gcast-play.on { border-color:var(--h3-accent); color:var(--h3-accent); }
.gcast-h { position:absolute; top:4px; width:11px; height:36px; margin-left:-5.5px; border-radius:3px;
  background:var(--gc-wave); cursor:ew-resize; box-shadow:0 0 0 1px #101010;
  touch-action:none; }
.gcast-track { touch-action:none; }
.gcast-quick { display:flex; gap:4px; flex-wrap:wrap; }
.gcast-quick button { all:unset; pointer-events:auto; cursor:pointer; padding:2px 7px; border-radius:4px;
  border:1px solid var(--h3-line); font-family:ui-monospace,Consolas,monospace;
  font-size:10px; color:var(--h3-dim); }
.gcast-quick button:hover { border-color:var(--h3-accent); color:var(--h3-accent); }
.gcast-quick button:focus-visible { outline:2px solid var(--h3-accent); outline-offset:1px; }
.gcast-h:focus-visible { outline:2px solid #fff; outline-offset:1px; }
.gcast-times { display:flex; align-items:center; gap:7px; justify-content:space-between;
  font-family:ui-monospace,Consolas,monospace; font-size:10px; color:var(--h3-dim); }
.gcast-chk { display:flex; align-items:center; gap:5px; font-size:11px; color:var(--h3-dim);
  cursor:pointer; user-select:none; }
.gcast-chk input { accent-color:var(--h3-accent); margin:0; }

/* prompt ------------------------------------------------------------ */
/* pWrap is a positioned parent so the gutter can be placed against it.
   MUST be declared before .gcast-promptlayer: equal specificity, later wins,
   and the expanded layer has to stay absolute or its top/left/right/bottom
   read as offsets and shove the prompt out of the node. */
.gcast-pwrap { position:relative; }
.gcast-promptlayer {
  position:absolute; left:10px; right:10px; bottom:10px; z-index:40;
  background:var(--h3-panel); border:1px solid var(--h3-line); border-radius:10px;
  padding:9px 11px 11px; display:flex; flex-direction:column; gap:6px;
  box-shadow:0 20px 50px #000c, 0 2px 10px #0008;
}
.gcast-promptlayer textarea { flex:1 1 auto; min-height:0; }
/* fl2va has only two slots, so the prompt takes the leftover room */
.gcast-promptfill { flex:1 1 auto; min-height:0; }
.gcast-promptfill textarea { flex:1 1 auto; }
.gcast-phead { display:flex; align-items:center; gap:8px; }

/* shot timeline: one thin band across the node standing for the whole clip,
   split where the [Shot N] markers say. Reading a shot list as timestamps
   buried in prose is the slow way to notice that shot 3 starts after the
   clip has ended. */
/* the prompt's left margin, colour-matched to the timeline segments */
/* clip-path as well as overflow: the dots are moved by a compositor-driven
   transform below, and clip-path clips a transformed descendant reliably. */
.gcast-gutter { position:absolute; width:6px; overflow:hidden; clip-path:inset(0);
  pointer-events:none; z-index:1; }
.gcast-gutter .inner { position:absolute; left:0; right:0; top:0; bottom:0; }
.gcast-gutter .dot { position:absolute; left:0; width:6px; height:6px; border-radius:50%; }

/* Zero-lag scroll sync where the browser supports it.
   A JS scroll listener always runs a frame after the compositor has already
   moved the text, which is the visible lag. Binding the transform to the
   textarea's own scroll timeline moves the dots in the SAME frame. Falls back
   to repainting from scrollTop when unsupported. */
.gcast-pwrap { timeline-scope: --gcastPrompt; }
.gcast-pwrap > textarea { scroll-timeline-name: --gcastPrompt; scroll-timeline-axis: y; }
.gcast-gutter .inner.sdriven { animation: gcast-gutter-scroll linear both;
  animation-timeline: --gcastPrompt; }
@keyframes gcast-gutter-scroll {
  from { transform: translateY(0); }
  to   { transform: translateY(var(--gcast-scroll-max, 0px)); }
}
/* offscreen twin of the textarea, used only to measure where each marker lands */
.gcast-mirror { position:absolute; left:-99999px; top:0; visibility:hidden;
  pointer-events:none; }
.gcast-shotbar { display:flex; flex-direction:column; gap:2px; }
.gcast-shotbar .ruler { position:relative; height:11px; }
.gcast-shotbar .ruler .t { position:absolute; bottom:0; width:1px;
  background:var(--h3-dim); opacity:.4; }
.gcast-shotbar .ruler .t.maj { opacity:.75; }
.gcast-shotbar .ruler .lbl { position:absolute; top:0; margin-left:3px; line-height:1;
  font-family:ui-monospace,Consolas,monospace; font-size:8px; color:var(--h3-dim); }
.gcast-shotbar .band { position:relative; display:flex; height:22px; width:100%;
  border-radius:5px; overflow:hidden; background:var(--h3-well); gap:1px; }
/* Wide grab area, thin visible line. 17px is easy to catch on a busy strip,
   but what you SEE stays a 2px seam - a 17px marker would cover the very
   segment edge you are trying to place. */
.gcast-shotbar .bnd { position:absolute; top:0; bottom:0; width:17px;
  margin-left:-8px; cursor:col-resize; touch-action:none; z-index:2; }
.gcast-shotbar .bnd::after { content:""; position:absolute; left:7px; top:0; bottom:0;
  width:2px; border-radius:1px; background:#fff; opacity:.3; transition:.12s; }
.gcast-shotbar .bnd:hover::after { opacity:.8; }
.gcast-shotbar .bnd.on::after { opacity:1; background:var(--h3-txt); }
.gcast-shotbar .seg { height:100%; min-width:0; overflow:hidden;
  display:flex; align-items:center; }
/* White ink, with a soft shadow so it survives the lighter colours in the
   cycle (the ochre and the sand) as well as the blues. */
.gcast-shotbar .seg .lab { min-width:0; padding:0 6px; font-size:10px; line-height:1;
  color:#fff; opacity:.96; text-shadow:0 1px 2px rgba(0,0,0,.55);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; pointer-events:none; }
.gcast-shotbar .seg.past { background-image:repeating-linear-gradient(
  45deg, #00000000 0 3px, #00000059 3px 6px); }
.gcast-shotbar .note { font-family:ui-monospace,Consolas,monospace; font-size:9.5px;
  color:var(--h3-dim); }
.gcast-shotbar .note.warn { color:#e0a35c; }
.gcast-pace { display:flex; align-items:center; gap:7px; }
.gcast-pace input[type="range"] { width:104px; accent-color:var(--h3-accent);
  pointer-events:auto; cursor:pointer; }
.gcast-ok { color:var(--gc-wave); }
.gcast-phead .spacer { margin-left:auto; }
.gcast-phead button { padding:2px 9px; font-size:10.5px; }
.gcast-chips { display:flex; flex-wrap:wrap; gap:5px; }
.gcast-chip { all:unset; pointer-events:auto; cursor:pointer; display:flex; align-items:center; gap:5px;
  background:var(--h3-raise); border:1px solid var(--h3-line); border-radius:20px;
  padding:2px 9px 2px 2px; font-family:ui-monospace,Consolas,monospace; font-size:10.5px;
  color:var(--h3-txt); transition:.13s; }
.gcast-chip:hover { border-color:var(--h3-accent); }
.gcast-chip.on { border-color:var(--h3-accent); background:var(--h3-accent-dim); }
.gcast-chip.on span:last-child { color:var(--h3-accent); }
.gcast-chip img, .gcast-chip video { width:18px; height:18px; border-radius:50%; object-fit:cover; }
.gcast-chip .glyph { width:18px; height:18px; border-radius:50%; background:var(--h3-bg);
  display:flex; align-items:center; justify-content:center; font-size:9px; color:var(--h3-accent); }
.gcast textarea { background:var(--h3-bg); color:var(--h3-txt); border:1px solid var(--h3-line);
  border-radius:7px; padding:8px 9px; font-family:inherit; font-size:12px; line-height:1.5;
  /* resize:none, not vertical. The browser's own grip writes an INLINE height
     onto the element, which outranks the flex fill and cannot be undone by
     anything the layout does - the panel then stays taller than the node
     however small the node is dragged, and the two come unstuck. The prompt
     already grows with the node; that is the handle. */
  resize:none; width:100%; box-sizing:border-box; outline:none;
  overflow-y:auto; overscroll-behavior:contain; }
/* preset bar + dialogs ------------------------------------------------ */
.gcast-bar { display:flex; align-items:center; gap:6px; }
.gcast-bar .name { font-family:ui-monospace,Consolas,monospace; font-size:11px;
  color:var(--h3-txt); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gcast-bar .name .dirty { color:var(--h3-dim); }
.gcast-bar .name .shot { color:#c9aeff; }
.gcast-bar .spacer { margin-left:auto; }
.gcast-btn { all:unset; pointer-events:auto; cursor:pointer; padding:4px 10px; border-radius:6px;
  border:1px solid var(--h3-line); background:var(--h3-panel);
  font-size:11px; color:var(--h3-txt); transition:.13s; }
.gcast-btn:hover { border-color:var(--h3-accent); color:var(--h3-accent); }
.gcast-btn:focus-visible { outline:2px solid var(--h3-accent); outline-offset:1px; }
.gcast-btn.ghost { background:transparent; color:var(--h3-dim); }

/* shots -------------------------------------------------------------
   Violet on purpose: amber already means "something is wrong" in the
   prompt check and the presentation strip, and blue/green are taken by
   the two weight families. */
.gcast-btn.shots { background:#2a2340; border-color:#584a78; color:#c9aeff; }
.gcast-btn.shots:hover { border-color:#a97bff; color:#e4d6ff; }
.gcast-shots {
  --h3-raise:#2b2b2b; --h3-line:#3b3b3b; --h3-well:#131313; --h3-panel:#212121;
  --h3-txt:#e3e3e3; --h3-dim:#979797; --h3-violet:#a97bff;
  --h3-accent:#58d1ff; --h3-accent-dim:#58d1ff26;
  font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  position:fixed; z-index:80; width:336px; box-sizing:border-box;
  background:var(--h3-raise); border:1px solid var(--h3-line); border-radius:10px;
  padding:8px; box-shadow:0 14px 34px #000c; color:var(--h3-txt); font-size:11.5px; }
.gcast-shots[data-mode="fl2va"] { --h3-accent:#59c14f; --h3-accent-dim:#59c14f26; }
.gcast-shots input { background:var(--h3-well); color:var(--h3-txt); box-sizing:border-box;
  border:1px solid var(--h3-line); border-radius:5px; padding:4px 7px; width:100%;
  font-family:inherit; font-size:11.5px; outline:none; }
.gcast-shots input:focus { border-color:var(--h3-violet); }
.gcast-shots-head { padding-bottom:7px; }
.gcast-shots-title { font-size:10px; letter-spacing:.08em; text-transform:uppercase;
  color:#c9aeff; padding:0 2px 5px; }
.gcast-shots-file { font-family:ui-monospace,Consolas,monospace; font-size:9.5px;
  color:var(--h3-dim); padding:5px 2px 0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gcast-shots-list { max-height:min(52vh, 420px); overflow:auto; display:flex; flex-direction:column; gap:4px; }
.gcast-shots-empty { color:var(--h3-dim); font-size:11px; padding:9px 4px; line-height:1.5; }
.gcast-shot { display:flex; align-items:center; gap:8px; padding:5px; border-radius:7px;
  border:1px solid transparent; cursor:pointer; }
.gcast-shot:hover { background:#ffffff0a; border-color:var(--h3-line); }
.gcast-shot.on { border-color:var(--h3-violet); background:#a97bff14; }
.gcast-shot .th { width:46px; height:30px; flex:0 0 auto; border-radius:4px; overflow:hidden;
  background:var(--h3-well); display:flex; align-items:center; justify-content:center; color:#4a4a4a; }
.gcast-shot .th img, .gcast-shot .th video { width:100%; height:100%; object-fit:cover; display:block; }
.gcast-shot .mid { min-width:0; flex:1 1 auto; }
.gcast-shot .nm { font-size:11.5px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.gcast-shot .meta { font-family:ui-monospace,Consolas,monospace; font-size:9.5px; color:var(--h3-dim); }
.gcast-shot .ctl { display:flex; gap:2px; flex:0 0 auto; opacity:0; transition:.13s; }
.gcast-shot:hover .ctl, .gcast-shot.on .ctl { opacity:1; }
.gcast-shot .ctl button { all:unset; pointer-events:auto; cursor:pointer; width:18px; height:18px;
  border-radius:4px; display:flex; align-items:center; justify-content:center;
  color:var(--h3-dim); font-size:9px; }
.gcast-shot .ctl button:hover { background:#ffffff14; color:var(--h3-txt); }
.gcast-shot .ctl button.rm:hover { background:#ff7a7a1a; color:#ff7a7a; }
.gcast-shots-foot { display:flex; align-items:center; gap:5px; flex-wrap:wrap;
  padding-top:8px; margin-top:8px; border-top:1px solid var(--h3-line); }
.gcast-shots-foot.proj { border-top:none; padding-top:5px; margin-top:0; }
.gcast-shots-foot .lbl { color:var(--h3-dim); font-size:10px; letter-spacing:.04em; }
.gcast-shots-foot .spacer { margin-left:auto; }
.gcast-shots-foot .gcast-btn { padding:3px 8px; font-size:10.5px; }
/* Revert is the rescue button: violet like the rest of the project row, and
   visibly dead when there is nothing to go back to. .gcast-btn is all:unset,
   so the disabled look has to be written out - the browser's own greying is
   reset away with everything else. */
.gcast-shots-foot .gcast-btn.revert { color:var(--h3-violet); border-color:#584a78; }
.gcast-shots-foot .gcast-btn.revert:hover { border-color:#a97bff; color:#e4d6ff; }
.gcast-shots-foot .gcast-btn[disabled] { opacity:.34; cursor:default; }
.gcast-shots-foot .gcast-btn[disabled]:hover { border-color:var(--h3-line); color:var(--h3-dim); }

.gcast-badge { font-family:ui-monospace,Consolas,monospace; font-size:9.5px;
  letter-spacing:.06em; padding:1px 6px; border-radius:10px; flex:0 0 auto;
  border:1px solid currentColor; }
.gcast-badge.ref { color:#58d1ff; }
.gcast-badge.fl { color:#59c14f; }

.gcast-modal {
  --h3-panel:#212121; --h3-raise:#2b2b2b; --h3-line:#3b3b3b; --h3-well:#131313;
  --h3-txt:#e3e3e3; --h3-dim:#979797; --h3-accent:#58d1ff;
  position:fixed; inset:0; z-index:9000; background:#000a;
  display:flex; align-items:center; justify-content:center;
  font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif; }
.gcast-sheet { background:var(--h3-panel); border:1px solid var(--h3-line);
  border-radius:12px; box-shadow:0 20px 60px #000c; width:460px; max-width:92vw;
  max-height:76vh; display:flex; flex-direction:column; overflow:hidden;
  color:var(--h3-txt); font-size:12px; }
.gcast-sheet h3 { margin:0; padding:13px 15px 10px; font-size:12px; font-weight:600;
  letter-spacing:.04em; border-bottom:1px solid var(--h3-line); }
.gcast-sheet .body { padding:13px 15px; display:flex; flex-direction:column; gap:9px;
  overflow:auto; }
.gcast-sheet .foot { padding:11px 15px; border-top:1px solid var(--h3-line);
  display:flex; gap:7px; justify-content:flex-end; }
.gcast-sheet input[type="text"] { background:var(--h3-well); color:var(--h3-txt);
  border:1px solid var(--h3-line); border-radius:6px; padding:8px 10px;
  font-size:12px; font-family:inherit; outline:none; width:100%; box-sizing:border-box; }
.gcast-sheet input[type="text"]:focus { border-color:var(--h3-accent); }
.gcast-sheet .hint { color:var(--h3-dim); font-size:11px; line-height:1.45; }
.gcast-row2 { display:flex; align-items:center; gap:9px; padding:8px 9px;
  border:1px solid var(--h3-line); border-radius:8px; cursor:pointer; background:var(--h3-well); }
.gcast-row2:hover { border-color:var(--h3-accent); }
.gcast-row2 .t { flex:1; min-width:0; }
.gcast-row2 .t b { display:block; font-weight:600; overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap; }
.gcast-row2 .t span { color:var(--h3-dim); font-size:10.5px;
  font-family:ui-monospace,Consolas,monospace; }
.gcast-row2 .del { all:unset; pointer-events:auto; cursor:pointer; color:var(--h3-dim); padding:2px 6px;
  border-radius:4px; }
.gcast-row2 .del:hover { color:#ff7a7a; background:#ff7a7a1a; }

.gcast-ac {
  --h3-raise:#2b2b2b; --h3-line:#3b3b3b; --h3-well:#131313;
  --h3-txt:#e3e3e3; --h3-dim:#979797;
  --h3-accent:#58d1ff; --h3-accent-dim:#58d1ff26;
  --gc-wave:#59c14f;
  font-family: ui-sans-serif, system-ui, "Segoe UI", sans-serif;
  position:absolute; z-index:70; background:var(--h3-raise);
  border:1px solid var(--h3-line); border-radius:8px; padding:4px; min-width:210px;
  max-height:236px; overflow:auto; box-shadow:0 10px 28px #000b; }
.gcast-ac[data-mode="fl2va"] { --h3-accent:#59c14f; --h3-accent-dim:#59c14f26; }
.gcast-ac button { all:unset; box-sizing:border-box; cursor:pointer; display:flex; width:100%;
  align-items:center; gap:8px; padding:5px 7px; border-radius:5px;
  font-family:ui-monospace,Consolas,monospace; font-size:11px; color:var(--h3-txt); }
.gcast-ac button[aria-selected="true"] { background:var(--h3-accent-dim); }
.gcast-ac img, .gcast-ac video { width:22px; height:22px; border-radius:50%; object-fit:cover; flex:0 0 auto; }
.gcast-ac .glyph { width:22px; height:22px; border-radius:50%; background:var(--h3-well);
  display:flex; align-items:center; justify-content:center; font-size:10px;
  color:var(--h3-accent); flex:0 0 auto; }
.gcast-ac .tag { margin-left:auto; color:var(--h3-dim); font-size:10px; }
.gcast-ac .glyph.mark { background:var(--h3-accent-dim); color:var(--h3-accent); }
.gcast-ac .none { padding:6px 8px; color:var(--h3-dim); font-size:11px; font-family:inherit; }
.gcast-pres { background:var(--h3-well); border:1px solid var(--h3-line); border-radius:7px;
  box-shadow:inset 0 1px 3px rgba(0,0,0,.38);
  padding:7px 9px; display:flex; flex-wrap:wrap; gap:4px 10px;
  font-family:ui-monospace,Consolas,monospace; font-size:10.5px; color:var(--h3-dim); }
.gcast-pres b { color:var(--h3-accent); font-weight:600; }
.gcast-pres .arrow { opacity:.4; }
.gcast-warn { color:#ffcc66; }
@media (prefers-reduced-motion: reduce) { .gcast * { transition:none !important; } }
`;

function injectCSS() {
  if (document.getElementById("gcast-css")) return;
  const s = document.createElement("style");
  s.id = "gcast-css"; s.textContent = CSS;
  document.head.appendChild(s);
}

/* ------------------------------------------------------------- helpers */

/* What is being dragged, read from the drag rather than the file: during a
 * dragover the name is not exposed, only the MIME type. Returns image, video,
 * audio, or null when it is not a file drag at all (a node, a link, text). */
function dragKind(e) {
  const dt = e.dataTransfer;
  if (!dt) return null;
  const types = Array.from(dt.types || []);
  if (!types.includes("Files")) return null;
  const item = Array.from(dt.items || []).find((i) => i.kind === "file");
  const mime = item?.type || "";
  if (mime.startsWith("image/")) return "image";
  if (mime.startsWith("video/")) return "video";
  if (mime.startsWith("audio/")) return "audio";
  return "";              /* a file, but the browser will not say what kind */
}

/* A dropdown that looks like the rest of the panel rather than an OS combo.
 * Same surface as the <select> it replaces: .value, .onchange, .disabled. */
function makeSelect(node, titleAttr) {
  const btn = el("button", "gcast-select");
  btn.type = "button";
  if (titleAttr) btn.title = titleAttr;
  const lbl = el("span", "lbl");
  btn.append(lbl, el("span", "caret", "\u25BE"));

  let opts = [], value = null, onchange = null, disabled = false, menu = null;
  let dismissing = false;      /* a press outside is waiting for its pointerup */

  const paint = () => {
    const o = opts.find((o) => o.value === value);
    lbl.textContent = o ? o.label : "";
    btn.setAttribute("aria-disabled", String(disabled));
  };

  const close = () => {
    dismissing = false;
    if (!menu) return;
    menu.remove();
    menu = null;
    document.removeEventListener("pointerdown", onOutside, true);
    document.removeEventListener("keydown", onKey, true);
    window.removeEventListener("wheel", close, true);
  };
  /* Dismiss on pointerUP, never on pointerdown. Removing the menu while the
   * button is still down means the matching pointerup lands on nothing, and
   * the v2 canvas is left believing a drag is in progress -- the graph then
   * pans with the mouse until the next press. Same reason items select on
   * click; this is the path that press missed. */
  const onOutside = (e) => {
    if (!menu || menu.contains(e.target) || dismissing) return;
    dismissing = true;
    const done = () => {
      window.removeEventListener("pointerup", done, true);
      window.removeEventListener("pointercancel", done, true);
      dismissing = false;
      close();
    };
    window.addEventListener("pointerup", done, true);
    window.addEventListener("pointercancel", done, true);
  };
  const onKey = (e) => { if (e.key === "Escape") { e.stopPropagation(); close(); } };

  const open = () => {
    close();
    menu = el("div", "gcast-menu");
    menu.dataset.mode = node?.h3ui?.mode || document.querySelector(".gcast")?.dataset.mode || "";
    let lastGroup = null;
    opts.forEach((o) => {
      if (o.group && o.group !== lastGroup) {
        lastGroup = o.group;
        menu.append(el("div", "grp", o.group));
      }
      const b = el("button", null);
      b.type = "button";
      b.append(el("span", null, o.label));
      if (o.note) b.append(el("span", "note", o.note));
      b.setAttribute("aria-selected", String(o.value === value));
      /* Select on CLICK, not pointerdown. Removing the menu on pointerdown
       * means the matching pointerup lands on nothing, and the v2 canvas is
       * left believing a drag is still in progress. */
      b.addEventListener("pointerdown", (e) => { e.stopPropagation(); });
      b.addEventListener("click", (e) => {
        e.preventDefault(); e.stopPropagation();
        close();
        if (o.value === value) return;
        value = o.value;
        paint();
        onchange?.(value);
      });
      menu.append(b);
    });
    document.body.append(menu);

    const r = btn.getBoundingClientRect();
    menu.style.minWidth = Math.max(r.width, 170) + "px";
    const h = menu.offsetHeight, w = menu.offsetWidth;
    menu.style.left = Math.max(6, Math.min(r.left, window.innerWidth - w - 8)) + "px";
    menu.style.top = (r.bottom + 4 + h > window.innerHeight - 8
      ? Math.max(6, r.top - h - 4)
      : r.bottom + 4) + "px";

    setTimeout(() => document.addEventListener("pointerdown", onOutside, true), 0);
    document.addEventListener("keydown", onKey, true);
    window.addEventListener("wheel", close, true);
  };

  btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); });
  btn.addEventListener("click", (e) => {
    e.preventDefault(); e.stopPropagation();
    if (disabled) return;
    menu ? close() : open();
  });

  return {
    el: btn,
    setOptions(list) { opts = list || []; paint(); },
    get value() { return value; },
    set value(v) { value = v; paint(); },
    set disabled(v) { disabled = !!v; paint(); },
    set onchange(fn) { onchange = fn; },
    close,
  };
}

const el = (tag, cls, txt) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (txt != null) n.textContent = txt;
  return n;
};

function viewURL(file) {
  if (!file) return "";
  const i = file.lastIndexOf("/");
  const sub = i >= 0 ? file.slice(0, i) : "";
  const name = i >= 0 ? file.slice(i + 1) : file;
  return api.apiURL(`/view?filename=${encodeURIComponent(name)}&subfolder=${encodeURIComponent(sub)}&type=input`);
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("image", file, file.name);
  body.append("subfolder", ASSET_SUBFOLDER);
  body.append("type", "input");
  const res = await api.fetchApi("/upload/image", { method: "POST", body });
  if (res.status !== 200) throw new Error(`upload failed (${res.status})`);
  const d = await res.json();
  return d.subfolder ? `${d.subfolder}/${d.name}` : d.name;
}

function pickFile(accept, multiple) {
  return new Promise((resolve) => {
    const inp = document.createElement("input");
    inp.type = "file"; inp.accept = accept;
    if (multiple) inp.multiple = true;
    inp.onchange = () => {
      const list = inp.files ? Array.from(inp.files) : [];
      resolve(multiple ? list : (list[0] || null));
    };
    inp.click();
  });
}

function probeDuration(url, isVideo) {
  return new Promise((resolve) => {
    const m = document.createElement(isVideo ? "video" : "audio");
    m.preload = "metadata";
    m.onloadedmetadata = () => resolve(Number.isFinite(m.duration) ? m.duration : 0);
    m.onerror = () => resolve(0);
    m.src = url;
  });
}

/* ================================================================= build */

function buildUI(node) {
  const root = el("div", "gcast");
  let st = blankState();
  let dataWidget = null;

  const commit = () => {
    if (!dataWidget) dataWidget = node.widgets?.find((w) => w.name === "h3_data");
    if (dataWidget) dataWidget.value = JSON.stringify(st);
    node.setDirtyCanvas(true, true);
  };

  /* ---- preset bar ---- */
  const bar = el("div", "gcast-bar");
  const nameLabel = el("div", "name");
  const bSave = el("button", "gcast-btn", "Save");
  const bSaveAs = el("button", "gcast-btn ghost", "Save as\u2026");
  const bPack = el("button", "gcast-btn ghost", "Save packed");
  const bLoad = el("button", "gcast-btn ghost", "Load");
  const bShots = el("button", "gcast-btn shots", "Project");
  bSave.title = "Write back over the file you last saved or opened";
  bSaveAs.title = "Choose a location. JSON, media referenced by filename";
  bPack.title = "Choose a location. Zip with the media inside \u2014 portable";
  bShots.title = "The clips of this project \u2014 switch between them, and save or open the project as one file";
  bar.append(nameLabel, el("div", "spacer"), bSave, bSaveAs, bPack, bLoad, bShots);

  /* ---- mode bar ---- */
  const modeBar = el("div", "gcast-modebar");
  const seg = el("div", "gcast-seg");
  const btnFL = el("button", null, "First / last");
  const btnRef = el("button", null, "Omni references");
  seg.append(btnFL, btnRef);
  const need = el("div", "gcast-need");
  modeBar.append(seg, need);

  /* ---- settings ---- */
  const row = el("div", "gcast-row");

  const cCanvas = el("div", "gcast-card");
  cCanvas.append(el("div", "gcast-label", "Canvas"));
  const canvasCtl = el("div", "gcast-ctl");
  const selRatio = makeSelect(node, "Aspect ratio");
  selRatio.setOptions(RATIOS.map((r) => ({ label: r.label, value: r.label, note: `${r.w} x ${r.h}` }))
    .concat([{ label: "Custom", value: "custom" }]));
  const selSize = makeSelect(node, "Canvas size for this ratio");
  const inW = el("input"); inW.type = "number"; inW.step = 32; inW.min = 32;
  const inH = el("input"); inH.type = "number"; inH.step = 32; inH.min = 32;
  canvasCtl.append(selRatio.el, selSize.el);
  const canvasCtl2 = el("div", "gcast-ctl");
  canvasCtl2.append(inW, el("span", "gcast-read", "×"), inH);
  cCanvas.append(canvasCtl2);
  const canvasNote = el("div", "gcast-read");
  cCanvas.append(canvasCtl);

  cCanvas.append(canvasNote);

  const cLen = el("div", "gcast-card");
  cLen.append(el("div", "gcast-label", "Length"));
  const lenCtl = el("div", "gcast-ctl");
  const selLen = makeSelect(node, "Clip length");
  selLen.setOptions(LENGTH_PRESETS.map((f) => ({
    label: fmtSecs(f / FPS), value: String(f), note: `${f} f`,
  })).concat([{ label: "Custom", value: "custom" }]));
  const inLen = el("input"); inLen.type = "number"; inLen.min = 5; inLen.step = 17;
  lenCtl.append(selLen.el, inLen);
  const lenNote = el("div", "gcast-read");
  cLen.append(lenCtl, lenNote);

  const cRef = el("div", "gcast-card");
  cRef.append(el("div", "gcast-label", "Reference size"));
  const tog = el("div", "gcast-toggle");
  const bMatch = el("button", null, "match");
  const bMax = el("button", "max", "max");
  tog.append(bMatch, bMax);
  const refNote = el("div", "gcast-read");
  cRef.append(tog, refNote);

  row.append(cCanvas, cLen, cRef);

  /* ---- slot areas ---- */
  const flWrap = el("div");
  flWrap.append(el("div", "gcast-label", "Keyframes"));
  const flGrid = el("div", "gcast-grid fl");
  flWrap.append(flGrid);

  const refWrap = el("div");
  const imgLabel = el("div", "gcast-label", "Reference images");
  const imgGrid = el("div", "gcast-grid img");
  const vidLabel = el("div", "gcast-label", "Reference videos");
  const vidGrid = el("div", "gcast-grid med");
  const audLabel = el("div", "gcast-label", "Reference audio");
  const audGrid = el("div", "gcast-grid med");
  refWrap.append(imgLabel, imgGrid, el("div", null, ""), vidLabel, vidGrid, el("div", null, ""), audLabel, audGrid);
  refWrap.style.display = "flex";
  refWrap.style.flexDirection = "column";
  refWrap.style.gap = "6px";

  /* ---- prompt ---- */
  const pWrap = el("div", "gcast-pwrap");
  pWrap.style.display = "flex"; pWrap.style.flexDirection = "column"; pWrap.style.gap = "6px";
  const pHead = el("div", "gcast-phead");
  pHead.append(el("div", "gcast-label", "Prompt"));
  const bExpand = el("button", "gcast-btn ghost", "⤡ Expand");
  bExpand.title = "Give the prompt the whole node";
  pHead.append(el("div", "spacer"), bExpand);
  const chips = el("div", "gcast-chips");
  const shotbar = el("div", "gcast-shotbar");
  const gutter = el("div", "gcast-gutter");
  const gutterInner = el("div", "inner");
  gutter.append(gutterInner);
  const ta = el("textarea");
  ta.placeholder = "Describe the clip. Click a reference above to drop its @tag in.";
  const sHead = el("div", "gcast-phead");
  sHead.append(el("div", "gcast-label", "Prompt check"));
  const speech = el("div", "gcast-pres");

  const presLabel = el("div", "gcast-label", "Sent to the encoder as");
  const pres = el("div", "gcast-pres");
  pWrap.append(pHead, chips, shotbar, ta, sHead, speech, presLabel, pres);
  pWrap.append(gutter);
  ta.style.paddingLeft = "17px";          // room for the gutter bar
  /* Only needed on the fallback path; the scroll timeline handles it natively. */
  ta.addEventListener("scroll", () => { if (!SCROLL_DRIVEN) paintGutterBars(); },
                      { passive: true });
  /* Expand, collapse, a node resize and the drag handle all change the
   * textarea's box; the gutter is positioned from it, so it has to follow. */
  if (window.ResizeObserver) {
    new ResizeObserver(() => syncGutter()).observe(ta);
  }
  window.addEventListener("resize", () => syncGutter());

  root.append(bar, modeBar, row, flWrap, refWrap, pWrap);

  /* ------------------------------------------------------------ slots */

  function clearSlot(slot) { for (const k of Object.keys(slot)) delete slot[k]; }

  /* A token is only "in the prompt" if it isn't a prefix of a longer one —
   * @video1 must not match inside @videoaudio1. */
  const tokenRe = (t) => new RegExp(t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "(?![0-9A-Za-z])", "g");
  const hasToken = (t) => !!t && tokenRe(t).test(st.prompt);

  function fileKind(f) {
    const t = f?.type || "";
    if (t.startsWith("image/")) return "image";
    if (t.startsWith("video/")) return "video";
    if (t.startsWith("audio/")) return "audio";
    return "";              /* unknown type: let it through rather than block */
  }

  /* First empty slot of the right kind, else the last one so a full rack still
     takes the file rather than dropping it on the floor. Tokens are renumbered
     against filled slots at emit, so the index chosen here does not matter. */
  async function routeToFreeSlot(kind, file) {
    const bank = kind === "image" ? st.slots.images
               : kind === "video" ? st.slots.videos
               : kind === "audio" ? st.slots.audios : null;
    if (!bank || !bank.length) return false;
    const slot = bank.find((sl) => !sl.file) || bank[bank.length - 1];
    await assign(slot, file, kind, null);
    return true;
  }

  function flashPanel() {
    root.classList.add("gcast-reject");
    setTimeout(() => root.classList.remove("gcast-reject"), 320);
  }

  function flashWrong(node_) {
    node_.classList.add("gcast-reject");
    setTimeout(() => node_.classList.remove("gcast-reject"), 320);
  }

  async function assign(slot, file, kind, token) {
    const name = await uploadFile(file);
    clearSlot(slot);
    slot.file = name;
    if (kind !== "image") {
      const d = await probeDuration(viewURL(name), kind === "video");
      slot.dur = d; slot.start = 0; slot.end = d;
      if (kind === "video") slot.audio = true;
    }
    render(); commit();
  }

  function wireDrop(node_, slot, kind, accept, token) {
    node_.addEventListener("click", async (e) => {
      if (e.target.closest(".gcast-x, .gcast-track, .gcast-chk, .gcast-times, .gcast-quick")) return;
      const f = await pickFile(accept);
      if (f) await assign(slot, f, kind, token);
    });
    node_.dataset.kind = kind;
    node_.addEventListener("dragover", (e) => {
      const k = dragKind(e);
      if (k === null) return;                    /* not a file: leave it alone */
      e.preventDefault(); e.stopPropagation();
      /* a mismatch stays unhighlighted, so the wrong slot never looks willing */
      node_.classList.toggle("drop", k === "" || k === kind);
    });
    /* dragleave also fires crossing into a child, which made the highlight
       flicker. Only clear when the pointer has really left the card. */
    node_.addEventListener("dragleave", (e) => {
      if (!node_.contains(e.relatedTarget)) node_.classList.remove("drop");
    });
    node_.addEventListener("drop", async (e) => {
      const k = dragKind(e);
      if (k === null) return;
      e.preventDefault(); e.stopPropagation(); node_.classList.remove("drop");
      const f = e.dataTransfer?.files?.[0];
      if (!f) return;
      if (fileKind(f) && fileKind(f) !== kind) { flashWrong(node_); return; }
      await assign(slot, f, kind, token);
    });
  }

  function imageSlot(slot, caption, token) {
    const card = el("div", "gcast-slot" + (slot.file ? " filled" : ""));
    const thumb = el("div", "gcast-thumb");
    if (slot.file) {
      const img = el("img"); img.src = viewURL(slot.file); img.loading = "lazy";
      thumb.append(img);
    } else thumb.append(el("div", "gcast-empty", "+"));
    const cap = el("div", "gcast-cap");
    cap.append(el("span", "n", caption));
    if (slot.file && token) cap.append(el("span", null, token));
    if (slot.file) {
      const x = el("button", "gcast-x", "×");
      x.title = "Clear slot";
      x.onclick = (e) => { e.stopPropagation(); clearSlot(slot); render(); commit(); };
      cap.append(x);
    }
    card.append(thumb, cap);
    return card;
  }

  /* ---- waveforms ---------------------------------------------------
   * Decoded once per file and cached. decodeAudioData handles whatever the
   * browser can play, so an mp4's soundtrack works as well as a wav.
   */
  const wavCache = new Map();

  function peaksFor(file) {
    if (wavCache.has(file)) return wavCache.get(file);
    const job = (async () => {
      const buf = await (await fetch(viewURL(file))).arrayBuffer();
      const Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) return null;
      const ctx = new Ctx();
      try {
        const audio = await ctx.decodeAudioData(buf);
        const ch = audio.getChannelData(0);
        const N = 1400, block = Math.max(1, Math.floor(ch.length / N));
        const out = new Float32Array(N);
        let peak = 0;
        for (let i = 0; i < N; i++) {
          let m = 0;
          const s0 = i * block;
          for (let k = 0; k < block; k += 3) {
            const v = Math.abs(ch[s0 + k] || 0);
            if (v > m) m = v;
          }
          out[i] = m;
          if (m > peak) peak = m;
        }
        if (peak > 0) for (let i = 0; i < N; i++) out[i] /= peak;   // normalise for display
        return out;
      } finally { ctx.close?.(); }
    })().catch(() => null);
    wavCache.set(file, job);
    return job;
  }

  function drawWave(cv, peaks) {
    const w = cv.clientWidth || 1, h = cv.clientHeight || 44;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.max(1, Math.round(w * dpr));
    cv.height = Math.max(1, Math.round(h * dpr));
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (!peaks) {
      g.strokeStyle = "rgba(255,255,255,.10)";
      g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke();
      return;
    }
    const mid = h / 2, pad = 3;
    g.fillStyle = "rgba(255,255,255,.34)";
    for (let x = 0; x < w; x++) {
      const v = peaks[Math.min(peaks.length - 1, Math.floor((x / w) * peaks.length))] || 0;
      const bar = Math.max(1, v * (mid - pad));
      g.fillRect(x, mid - bar, 1, bar * 2);
    }
  }

  /* Reference videos are re-snapped DOWN to the 17k+5 grid after trimming, so a
   * 20-frame trim silently becomes 5. Handles stay free; the readout tells the
   * truth about what survives. */
  const gridDown = (n) => { let g = Math.floor(n); while (g > 0 && g % 17 !== 5) g--; return g; };
  const effFrames = (span, capFrames) => {
    let want = Math.max(0, Math.round(span * FPS) + 1);
    if (capFrames) want = Math.min(want, capFrames);
    return gridDown(want);
  };
  const gridSpans = (dur) => {
    const out = [];
    for (let g = 5; ; g += 17) {
      const sp = (g - 1) / FPS;
      if (sp > dur) break;
      out.push({ g, sp });
    }
    return out;
  };

  /* dual-handle trim, in seconds, clamped to the clip's own duration */
  function trim(slot, usedSeconds, isVideo, media, label, tailOnly) {
    const wrap = el("div", "gcast-trim");
    const track = el("div", "gcast-track");
    const wave = el("canvas", "gcast-wave");
    const head = el("div", "gcast-head");
    const span = el("div", "gcast-span");
    const hA = el("div", "gcast-h"); hA.tabIndex = 0; hA.title = "Trim start";
    const hB = el("div", "gcast-h"); hB.tabIndex = 0; hB.title = "Trim end";
    const used = el("div", "gcast-used");
    track.append(wave, span, used, head, hA, hB);
    if (label) track.append(el("div", "gcast-wavlabel", label));
    const times = el("div", "gcast-times");
    const play = el("button", "gcast-play", "\u25B6");
    play.title = "Play the trimmed range";
    const tL = el("span"), tR = el("span");
    times.append(play, tL, tR);
    wrap.append(track, times);

    /* waveform: draw once decoded, and again if the node is resized */
    if (slot.file) {
      peaksFor(slot.file).then((pk) => drawWave(wave, pk));
      try {
        const ro = new ResizeObserver(() => {
          peaksFor(slot.file).then((pk) => drawWave(wave, pk));
        });
        ro.observe(track);
      } catch (e) { /* older browser: static draw is fine */ }
    }

    /* transport: play only what is selected, and stop at the out point */
    let raf = 0;
    const stopAt = () => {
      if (!media || media.paused) return false;
      /* the out point is enforced here rather than on timeupdate, which
       * only fires ~4x a second and overshoots audibly */
      if (media.currentTime >= slot.end - 0.005 || media.currentTime < slot.start - 0.05) {
        media.pause();
        media.currentTime = slot.start;
        return true;
      }
      return false;
    };
    const paintHead = () => {
      if (!media || media.paused || !dur) { head.style.display = "none"; return; }
      if (stopAt()) { head.style.display = "none"; return; }
      head.style.display = "";
      head.style.left = (media.currentTime / dur) * 100 + "%";
      raf = requestAnimationFrame(paintHead);
    };
    if (media) {
      media.addEventListener("timeupdate", stopAt);
      media.addEventListener("play", () => { play.textContent = "\u23F8"; play.classList.add("on"); paintHead(); });
      media.addEventListener("pause", () => {
        play.textContent = "\u25B6"; play.classList.remove("on");
        cancelAnimationFrame(raf); head.style.display = "none";
      });
      play.onclick = (e) => {
        e.stopPropagation();
        if (!media.paused) { media.pause(); return; }
        media.muted = false;
        media.currentTime = slot.start;
        media.play().catch(() => {});
      };
    } else {
      play.disabled = true;
      play.style.opacity = .35;
      play.title = "No audio on this clip";
    }

    const dur = slot.dur || 0;
    const capFrames = usedSeconds != null ? Math.round(usedSeconds * FPS) : null;

    if (!Number.isFinite(slot.start) || slot.start < 0 || slot.start >= dur) slot.start = 0;
    if (!Number.isFinite(slot.end) || slot.end > dur || slot.end <= slot.start) slot.end = dur;

    const paint = () => {
      if (!dur) return;
      const a = (slot.start / dur) * 100, b = (slot.end / dur) * 100;
      span.style.left = a + "%"; span.style.width = Math.max(0, b - a) + "%";
      hA.style.left = a + "%"; hB.style.left = b + "%";
      tL.textContent = isVideo
        ? `${fmtSecs(slot.start)}  f${Math.round(slot.start * FPS)}`
        : fmtSecs(slot.start);
      const len = slot.end - slot.start;

      if (!isVideo) {
        tR.textContent = fmtSecs(len);
        tR.className = "";
        used.style.display = "none";
        return;
      }
      const f = effFrames(len, capFrames);
      const clipped = capFrames != null && Math.round(len * FPS) + 1 > capFrames;
      tR.textContent = f < 5 ? "too short - needs 5 frames" : `${f} f used  (${fmtSecs(len)} trimmed)`;
      tR.className = (f < 5 || clipped) ? "gcast-warn" : "";
      if (clipped) {
        used.style.display = "";
        used.style.left = ((slot.start + capFrames / FPS) / dur) * 100 + "%";
      } else used.style.display = "none";
    };
    paint();

    /* Park the video on a given time so the thumbnail shows the frame you are
     * pointing at. Hunting for a readable 5-22f window is done by eye, so the
     * picture has to follow the window, not the other way round. Coalesced to
     * one seek per frame - seeking on every pointermove stutters. */
    let seekReq = 0;
    const seekPreview = (t) => {
      if (!media || media.tagName !== "VIDEO") return;
      cancelAnimationFrame(seekReq);
      seekReq = requestAnimationFrame(() => {
        try {
          if (!media.paused) media.pause();
          media.currentTime = Math.min(Math.max(0, t), Math.max(0, dur - 1 / FPS));
        } catch (_) { /* not seekable yet */ }
      });
    };

    const drag = (handle, isStart) => {
      let dragging = false;

      const apply = (clientX) => {
        const r = track.getBoundingClientRect();
        let t = ((clientX - r.left) / r.width) * dur;
        t = Math.min(dur, Math.max(0, t));
        if (isStart) slot.start = Math.min(t, slot.end - 0.25);
        else slot.end = Math.max(t, slot.start + 0.25);
        paint();
        seekPreview(isStart ? slot.start : slot.end);
      };

      /* LiteGraph's canvas consumes pointer/mouse moves, so listen in the
       * CAPTURE phase on document and take pointer capture on the handle. */
      const move = (e) => {
        if (!dragging) return;
        e.preventDefault(); e.stopPropagation();
        apply(e.clientX);
      };
      const up = (e) => {
        if (!dragging) return;
        dragging = false;
        try { handle.releasePointerCapture(e.pointerId); } catch (_) {}
        document.removeEventListener("pointermove", move, true);
        document.removeEventListener("pointerup", up, true);
        document.removeEventListener("mousemove", moveMouse, true);
        document.removeEventListener("mouseup", up, true);
        commit();
      };
      const moveMouse = (e) => { if (dragging) apply(e.clientX); };

      handle.addEventListener("pointerdown", (e) => {
        if (!dur) return;
        e.preventDefault(); e.stopPropagation();
        dragging = true;
        try { handle.setPointerCapture(e.pointerId); } catch (_) {}
        document.addEventListener("pointermove", move, true);
        document.addEventListener("pointerup", up, true);
        document.addEventListener("mousemove", moveMouse, true);
        document.addEventListener("mouseup", up, true);
      });
      handle.addEventListener("pointermove", move);
      handle.addEventListener("keydown", (e) => {
        const step = e.shiftKey ? 1 : 1 / FPS;
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault();
        const d = e.key === "ArrowLeft" ? -step : step;
        if (isStart) slot.start = Math.min(Math.max(0, slot.start + d), slot.end - 0.25);
        else slot.end = Math.max(Math.min(dur, slot.end + d), slot.start + 0.25);
        paint(); commit();
      });
    };
    /* A continuation window is not free to sit anywhere: it has to END on the
     * previous clip's last frame or the new clip continues from the wrong
     * instant, and the two pieces are then not adjacent in time. Length is the
     * only choice, so the handles and the slide are off in that mode. */
    if (!tailOnly) { drag(hA, true); drag(hB, false); }
    else { hA.style.display = "none"; hB.style.display = "none"; }

    /* Slide the whole window by dragging the track, length preserved.
     *
     * This is bound to the TRACK, not to the span. At the sizes that matter
     * here a 5f window is a few pixels wide and sits entirely underneath the
     * two 11px handles, so the span itself is not a grabbable target. Landing
     * inside the current window keeps the grab offset so the block moves under
     * the finger; landing outside jumps the window to the pointer.
     *
     * The handles stopPropagation on pointerdown, so resizing still wins over
     * sliding. Same capture-phase document listeners as the handles, for the
     * same reason - LiteGraph's canvas eats the move stream otherwise. */
    if (dur && !tailOnly) {
      track.classList.add("slidable");
      let sliding = false, grabOff = 0;

      const timeAt = (clientX) => {
        const r = track.getBoundingClientRect();
        return Math.min(dur, Math.max(0, ((clientX - r.left) / r.width) * dur));
      };
      const place = (clientX) => {
        const len = slot.end - slot.start;
        let s = timeAt(clientX) - grabOff;
        s = Math.min(Math.max(0, s), Math.max(0, dur - len));
        slot.start = s;
        slot.end = Math.min(dur, s + len);
        paint();
        seekPreview(slot.start);
      };

      const sMove = (e) => {
        if (!sliding) return;
        e.preventDefault(); e.stopPropagation();
        place(e.clientX);
      };
      const sMoveMouse = (e) => { if (sliding) place(e.clientX); };
      const sUp = (e) => {
        if (!sliding) return;
        sliding = false;
        track.classList.remove("sliding");
        try { track.releasePointerCapture(e.pointerId); } catch (_) {}
        document.removeEventListener("pointermove", sMove, true);
        document.removeEventListener("pointerup", sUp, true);
        document.removeEventListener("mousemove", sMoveMouse, true);
        document.removeEventListener("mouseup", sUp, true);
        commit();
      };

      track.addEventListener("pointerdown", (e) => {
        e.preventDefault(); e.stopPropagation();
        const t = timeAt(e.clientX);
        grabOff = (t >= slot.start && t <= slot.end) ? t - slot.start : 0;
        sliding = true;
        track.classList.add("sliding");
        try { track.setPointerCapture(e.pointerId); } catch (_) {}
        document.addEventListener("pointermove", sMove, true);
        document.addEventListener("pointerup", sUp, true);
        document.addEventListener("mousemove", sMoveMouse, true);
        document.addEventListener("mouseup", sUp, true);
        place(e.clientX);
      });

      /* Arrow keys nudge the window a frame at a time, Shift a second. */
      track.tabIndex = 0;
      track.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault(); e.stopPropagation();
        const len = slot.end - slot.start;
        const d = (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 1 : 1 / FPS);
        const s = Math.min(Math.max(0, slot.start + d), Math.max(0, dur - len));
        slot.start = s; slot.end = Math.min(dur, s + len);
        paint(); seekPreview(slot.start); commit();
      });
    }

    /* Span picks set an exact grid length - the values that survive the 17k+5
     * re-snap, so nothing is silently thrown away.
     *
     * An untouched clip (still the full range) gets the tail, which is what a
     * chunk continuation wants. Once the window has been moved the pick keeps
     * the position and only changes the length, because the expensive part is
     * finding the moment and re-finding it after every size change was the
     * annoyance. "> end" pins to the tail on demand. */
    if (isVideo && dur) {
      const quick = el("div", "gcast-quick");
      const untouched = () => slot.start <= 0.001 && slot.end >= dur - 0.001;
      gridSpans(dur).slice(0, 5).forEach((r) => {
        const b = el("button", null, `${r.g}f`);
        b.title = tailOnly
          ? `carry over the last ${r.g} frames (${fmtSecs(r.sp)})`
          : `${r.g} frames (${fmtSecs(r.sp)}) \u2014 drag the track to slide the window`;
        b.onclick = (e) => {
          e.stopPropagation();
          slot.start = (tailOnly || untouched())
            ? Math.max(0, dur - r.sp)
            : Math.min(slot.start, Math.max(0, dur - r.sp));
          slot.end = Math.min(dur, slot.start + r.sp);
          paint(); seekPreview(slot.start); commit();
        };
        quick.append(b);
      });
      const bEnd = el("button", null, "\u203A end");
      bEnd.title = "Move the window to the clip's end, same length";
      bEnd.onclick = (e) => {
        e.stopPropagation();
        const len = slot.end - slot.start;
        slot.end = dur;
        slot.start = Math.max(0, dur - len);
        paint(); seekPreview(slot.start); commit();
      };
      if (!tailOnly) quick.append(bEnd);   // a tail window is always at the end
      wrap.append(quick);
    }
    return wrap;
  }

  function mediaSlot(slot, kind, caption, token, usedSeconds, opts) {
    opts = opts || {};
    const card = el("div", "gcast-media" + (slot.file ? " filled" : "")
                    + (opts.cont ? " cont" : ""));
    const cap = el("div", "gcast-cap");
    if (opts.cont) {
      /* An arrow entering a box: this is where the clip comes IN from, not a
         reference standing beside it. */
      const ico = el("span", "gcast-ico");
      ico.innerHTML = '<svg viewBox="0 0 24 24" width="11" height="11" '
        + 'fill="currentColor" aria-hidden="true">'
        + '<path d="M13 3h7a1 1 0 0 1 1 1v16a1 1 0 0 1-1 1h-7v-2h6V5h-6V3z"/>'
        + '<path d="M11 7l5 5-5 5v-3H3v-4h8V7z"/></svg>';
      cap.append(ico);
    }
    cap.append(el("span", "n", caption));
    if (slot.file && token) cap.append(el("span", null, token));
    if (slot.file) {
      const x = el("button", "gcast-x", "×");
      x.title = "Clear slot";
      x.onclick = (e) => { e.stopPropagation(); clearSlot(slot); render(); commit(); };
      cap.append(x);
    }

    let body, media = null;
    if (kind === "video") {
      body = el("div", "gcast-thumb");
      if (slot.file) {
        const v = el("video");
        media = v;
        v.src = viewURL(slot.file); v.muted = true; v.playsInline = true; v.preload = "metadata";
        v.loop = false;   // looping wraps to frame 0 and sails past the out point
        // silent hover preview, but only while the transport is not in use
        v.onmouseenter = () => { if (v.muted) v.play().catch(() => {}); };
        v.onmouseleave = () => { if (v.muted) { v.pause(); v.currentTime = slot.start || 0; } };
        body.append(v);
      } else body.append(el("div", "gcast-empty", "+"));
    } else {
      if (slot.file) {
        media = el("audio");
        media.src = viewURL(slot.file);
        media.preload = "metadata";
        media.style.display = "none";
        card.append(media);
        body = null;      // the waveform carries the filename instead
      } else {
        body = el("div", "gcast-wav", "+  audio");
      }
    }

    card.append(cap);
    if (body) card.append(body);
    if (slot.file && slot.dur) {
      const label = (kind === "audio") ? slot.file.split("/").pop() : null;
      card.append(trim(slot, usedSeconds, kind === "video", media, label,
                       !!opts.tailOnly));
    }
    if (slot.file && kind === "video") {
      const lab = el("label", "gcast-chk");
      const cb = el("input"); cb.type = "checkbox"; cb.checked = !!slot.audio;
      cb.onchange = () => { slot.audio = cb.checked; render(); commit(); };
      lab.append(cb, el("span", null, "send its sound too"));
      card.append(lab);
    }

    /* Levelling the carried frames onto their own last frame, before they are
     * encoded. Measured on his renders: the model continues from the window's
     * AVERAGE exposure, not its final frame, so a window ending darker than
     * its average produced a brighter clip and vice versa. Off by default -
     * with a true tail window the drift measured near zero, so this is for the
     * case where it does not. */
    if (slot.file && opts.cont) {
      const lab2 = el("label", "gcast-chk");
      const cb2 = el("input"); cb2.type = "checkbox"; cb2.checked = !!slot.flatten;
      cb2.onchange = () => { slot.flatten = cb2.checked; render(); commit(); };
      lab2.append(cb2, el("span", null, "level exposure to the last frame"));
      lab2.title = "Only if the extension comes back brighter or darker than the "
                 + "clip it continues from";
      card.append(lab2);
    }

    wireDrop(body || card, slot, kind, kind === "video" ? "video/*" : "audio/*", token);
    return card;
  }

  /* ----------------------------------------------------------- render */

  function render() {
    try { renderInner(); }
    catch (e) { console.error("[H3 Studio] render failed:", e); }
  }

  function renderInner() {
    root.dataset.mode = st.mode;
    btnFL.setAttribute("aria-pressed", String(st.mode === "fl2va"));
    btnRef.setAttribute("aria-pressed", String(st.mode === "ref2va"));
    need.innerHTML = st.mode === "fl2va"
      ? 'load <b>minimax_h3_fl2va_*</b>'
      : 'load <b>minimax_h3_ref2va_*</b>';

    const fam = findRatio(st.width, st.height);
    selRatio.value = fam ? fam.label : "custom";
    if (fam) {
      selSize.setOptions(fam.sizes.map((z, i) => ({
        label: `${z.w} x ${z.h}`,
        value: `${z.w}x${z.h}`,
        note: i === 0 ? "native" : (z.w * z.h <= 384 * 640 ? "draft" : ""),
      })));
      selSize.value = `${st.width}x${st.height}`;
      selSize.disabled = false;
    } else {
      selSize.setOptions([{ label: "custom size", value: "custom" }]);
      selSize.value = "custom";
      selSize.disabled = true;
    }
    inW.value = st.width; inH.value = st.height;
    const mp = (st.width * st.height / 1e6).toFixed(2);
    canvasNote.innerHTML = `<b>${mp} MP</b> · ${(st.width / st.height).toFixed(3)}:1`
      + (st.width * st.height > 768 * 1344 ? ` · <span class="gcast-warn">past H3's area cap</span>` : "");

    const frames = alignFrames(st.length);
    selLen.value = LENGTH_PRESETS.includes(frames) ? String(frames) : "custom";
    inLen.value = frames;
    lenNote.innerHTML = `<b>${frames} frames</b> · ${fmtSecs(frames / FPS)} at 24 fps`;

    bMatch.setAttribute("aria-pressed", String(st.ref_image_size === "match"));
    bMax.setAttribute("aria-pressed", String(st.ref_image_size === "max"));
    tog.classList.toggle("warn", st.ref_image_size === "max");
    refNote.textContent = st.ref_image_size === "match"
      ? "scaled to the canvas — faster"
      : "2048px short edge — stronger identity, slower";

    const isFL = st.mode === "fl2va";
    /* fl2va already gives the prompt the leftover room, so Expand has nothing
     * to add there -- hide it and ignore any stored expanded state. */
    const expanded = !isFL && !!node.properties?.gcast_prompt_big;
    bExpand.style.display = isFL ? "none" : "";
    /* Spare height always goes to the prompt rather than sitting empty at the
     * bottom -- so a taller node just means more room to write. */
    const fill = !expanded;
    pWrap.classList.toggle("gcast-promptfill", fill);
    /* Clear any inline height left behind by the old resize grip - a workflow
     * saved while the textarea had been dragged would otherwise stay pinned
     * tall forever, since an inline height beats the flex fill. */
    if (ta.style.height) ta.style.height = "";
    ta.style.minHeight = expanded ? "0px" : (isFL ? PROMPT_H_FL : PROMPT_H_REF) + "px";
    bExpand.textContent = expanded ? "⤢ Collapse" : "⤡ Expand";
    pWrap.classList.toggle("gcast-promptlayer", expanded);
    root.style.minHeight = contentH + "px";
    if (expanded) {
      placeLayer();
    } else {
      pWrap.style.top = "";
      measureContent();
    }
    flWrap.style.display = isFL ? "" : "none";
    refWrap.style.display = isFL ? "none" : "flex";
    cRef.style.opacity = isFL ? "0.35" : "1";
    cRef.style.pointerEvents = isFL ? "none" : "";

    const { rows, tags } = presentation(st);

    flGrid.replaceChildren();
    if (isFL) {
      [["first", "First frame", "@first"], ["last", "Last frame", "@last"]].forEach(([key, cap, tok]) => {
        const slot = st.slots[key];
        const card = imageSlot(slot, cap, tok);
        wireDrop(card, slot, "image", "image/*", tok);
        flGrid.append(card);
      });
    }

    const usedSeconds = frames / FPS;
    imgGrid.replaceChildren();
    vidGrid.replaceChildren();
    audGrid.replaceChildren();
    if (!isFL) {
      st.slots.images.forEach((slot, i) => {
        const card = imageSlot(slot, String(i + 1), `@image${i + 1}`);
        wireDrop(card, slot, "image", "image/*", `@image${i + 1}`);
        imgGrid.append(card);
      });
      st.slots.videos.forEach((slot, i) => vidGrid.append(mediaSlot(slot, "video", `video ${i + 1}`, `@video${i + 1}`, usedSeconds)));
      st.slots.audios.forEach((slot, i) => audGrid.append(mediaSlot(slot, "audio", `audio ${i + 1}`, `@audio${i + 1}`, null)));
    }

    /* A freshly dropped clip arrives as its full range, which is never what a
     * continuation wants. Snap it to the last 22 frames - the tested window -
     * or to the largest grid run the clip can actually supply. */
    const c = st.cont;
    if (c.file && c.dur && (c.start || 0) <= 0.001 && (c.end || 0) >= c.dur - 0.001) {
      const spans = gridSpans(c.dur);
      const pick = spans.find((r) => r.g === 22) || spans[spans.length - 1];
      if (pick) { c.end = c.dur; c.start = Math.max(0, c.dur - pick.sp); }
    }
    /* No section of its own: it joins the row that is already on screen, so the
     * node stays exactly as tall as it was. Both modes get it - continuing is
     * orthogonal to first/last vs omni. */
    const contCard = mediaSlot(c, "video", "continue from", null, usedSeconds,
                               { tailOnly: true, cont: true });
    if (isFL) { flGrid.className = "gcast-grid fl3"; flGrid.append(contCard); }
    else { vidGrid.className = "gcast-grid med4"; vidGrid.append(contCard); }

    renderTags();
    renderCheck();
    paintPresetName();
    node.h3MinHeight = st.mode === "fl2va" ? MIN_H_FL : MIN_H_REF;
  }

  function renderCheck() {
    const rows = promptCheck(st.prompt || "", alignFrames(st.length) / FPS);
    speech.replaceChildren();

    let flagged = 0;
    rows.forEach((r) => {
      if (r.state === "warn") flagged++;
      const item = el("span");
      item.innerHTML = `<b>${r.label}</b> <span class="${r.state === "warn" ? "gcast-warn" : "gcast-ok"}">${r.text}</span>`;
      speech.append(item);
    });

    if (flagged) {
      speech.append(el("span", "gcast-warn",
        "\u00b7 H3 invents speech to fill whatever the prompt leaves unsaid"));
    }
    renderShotBar();
  }

  /* Colours cycle and mean nothing but "next shot" -- deliberately away from
   * the accent (weight family) and violet (project), which do mean something. */
  const SHOT_COLOURS = ["#4f8cd6", "#4fae8b", "#c9a24a", "#9a72c9", "#57a0b8", "#d08f6a"];

  /* Rewrite one marker's timestamp in the prompt. The whole point of the drag
   * is that it edits the text you will actually send - a timeline that only
   * moved a picture around would be decoration. */
  function writeShotTime(mark, seconds) {
    const stamp = `At ${mmss(Math.max(0, seconds))}`;
    const p = st.prompt || "";
    const caret = ta.selectionStart;
    const scroll = ta.scrollTop;   // assigning .value can reset it
    let out, delta, at;
    if (mark.tsFrom != null) {
      out = p.slice(0, mark.tsFrom) + stamp + p.slice(mark.tsTo);
      delta = stamp.length - (mark.tsTo - mark.tsFrom);
      at = mark.tsFrom;
    } else {
      const ins = ` ${stamp},`;
      out = p.slice(0, mark.afterBracket) + ins + p.slice(mark.afterBracket);
      delta = ins.length;
      at = mark.afterBracket;
    }
    st.prompt = out;
    ta.value = out;
    ta.scrollTop = scroll;
    const pos = caret > at ? Math.max(0, caret + delta) : caret;
    try { ta.setSelectionRange(pos, pos); } catch (_) {}
    renderTags(); commit(); renderCheck();
  }

  /* The opening words of a shot, for the label inside its segment. Trimmed of
   * the comma and spaces left over from the timestamp; CSS does the actual
   * truncation, so a wide segment shows more than a narrow one. */
  function shotBlurb(prompt, s) {
    let body = String(prompt || "").slice(s.bodyFrom, s.bodyTo == null ? undefined : s.bodyTo);
    body = body.replace(/^[\s,;:.\-]+/, "").replace(/\s+/g, " ").trim();
    return body;
  }

  /* Colour bar in the prompt's left margin, matching the timeline segment.
   *
   * A textarea cannot style a range of its own text, so the vertical extent of
   * each [Shot N] block is measured in a hidden MIRROR div that copies the
   * textarea's font, width, padding and wrapping. A zero-width span dropped at
   * each marker index reports the line it landed on; that is where the bar
   * starts. The gutter then only has to scroll with the textarea. */
  let mirror = null;
  const MIRROR_PROPS = ["fontFamily", "fontSize", "fontWeight", "fontStyle", "fontVariant",
    "letterSpacing", "lineHeight", "textTransform", "wordSpacing", "textIndent",
    "whiteSpace", "wordBreak", "overflowWrap", "tabSize"];

  function shotTops(marks) {
    if (!mirror) { mirror = el("div", "gcast-mirror"); document.body.append(mirror); }
    const cs = getComputedStyle(ta);
    MIRROR_PROPS.forEach((k) => { mirror.style[k] = cs[k]; });

    /* Padding is NOT copied. Copying it made every measured position depend on
     * the mirror reproducing the textarea's box exactly, and any mismatch
     * showed up as a constant shift of every bar. Instead the mirror is pure
     * content - zero padding, content-box, the textarea's inner width - and
     * the padding is added back explicitly, once, below. */
    const padL = parseFloat(cs.paddingLeft) || 0;
    const padR = parseFloat(cs.paddingRight) || 0;
    const padT = parseFloat(cs.paddingTop) || 0;
    mirror.style.padding = "0";
    mirror.style.boxSizing = "content-box";
    mirror.style.width = Math.max(10, ta.clientWidth - padL - padR) + "px";

    const text = st.prompt || "";
    mirror.replaceChildren();
    const spans = [];
    let pos = 0;
    for (const m of marks) {
      if (m.idx == null || m.idx < pos) continue;
      mirror.append(document.createTextNode(text.slice(pos, m.idx)));
      const sp = document.createElement("span");
      sp.textContent = "\u200b";
      mirror.append(sp);
      spans.push(sp);
      pos = m.idx;
    }
    mirror.append(document.createTextNode(text.slice(pos) + "\n"));
    const tops = spans.map((sp) => sp.offsetTop + padT);
    return { tops, height: mirror.scrollHeight + padT };
  }

  let lastGutter = null;
  let gutterState = null;   // { tops, height, lineHeight, colours } from the last measure
  const SCROLL_DRIVEN = typeof CSS !== "undefined" && !!CSS.supports
    && CSS.supports("animation-timeline: --x")
    && CSS.supports("scroll-timeline-name: --x");

  function syncGutter() {
    if (lastGutter) renderGutter(lastGutter.marks, lastGutter.colour);
  }

  /* Place the bars for the current scroll offset.
   *
   * Every bar is CLAMPED in script to the visible band rather than left to
   * overflow:hidden. Relying on the clip meant a bar could be drawn far above
   * the box and show up over the timeline if anything about the containing
   * block was not what I assumed. Clamping cannot do that: nothing outside
   * 0..clientHeight is ever written. */
  /* A dot per shot rather than a bar spanning it.
   *
   * A full-height bar has to be right at BOTH ends, so every reflow, rewrite
   * and scroll was another chance to be a few pixels out - and being out is
   * obvious when an edge is meant to line up with a paragraph. A dot only has
   * to sit beside the right line, which the same measurement gives easily. */
  function paintGutterBars() {
    if (!gutterState) return;
    const view = ta.clientHeight;
    const lh = gutterState.lineHeight || 18;
    gutterInner.replaceChildren();

    if (SCROLL_DRIVEN) {
      /* Dots sit at CONTENT coordinates and the whole layer is slid by the
       * scroll timeline, so nothing has to be recomputed while scrolling. */
      gutterInner.classList.add("sdriven");
      const span = Math.max(0, ta.scrollHeight - view);
      gutterInner.style.setProperty("--gcast-scroll-max", (-span) + "px");
      gutterState.tops.forEach((top, i) => {
        const dot = el("div", "dot");
        dot.style.top = (top + lh / 2 - 3) + "px";
        dot.style.background = gutterState.colours[i];
        gutterInner.append(dot);
      });
      return;
    }

    const off = ta.scrollTop;
    gutterState.tops.forEach((top, i) => {
      const y = top - off + lh / 2;               // centre of the marker's line
      if (y < 4 || y > view - 4) return;          // scrolled out
      const dot = el("div", "dot");
      dot.style.top = (y - 3) + "px";
      dot.style.background = gutterState.colours[i];
      gutterInner.append(dot);
    });
  }

  function renderGutter(marks, colourFor, deferred) {
    lastGutter = { marks, colour: colourFor };
    /* Expand re-flows the textarea a frame later, so a paint taken now would
     * measure a zero-height box and the bars would vanish. Wait for a real
     * size instead of drawing into nothing. */
    if (!ta.clientHeight) { requestAnimationFrame(syncGutter); return; }

    const cs = getComputedStyle(ta);
    gutter.style.top = (ta.offsetTop + parseFloat(cs.borderTopWidth || 0)) + "px";
    gutter.style.left = (ta.offsetLeft + parseFloat(cs.borderLeftWidth || 0) + 6) + "px";
    gutter.style.height = ta.clientHeight + "px";

    const lineHeight = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.5 || 18;
    if (!marks.length) {
      gutterState = { tops: [0], height: ta.scrollHeight, lineHeight, colours: [colourFor(0)] };
    } else {
      const { tops, height } = shotTops(marks);
      gutterState = { tops, height, lineHeight, colours: tops.map((_, i) => colourFor(i)) };
    }
    paintGutterBars();
    /* One re-measure a frame later, never more: the first pass can land before
     * the textarea has re-wrapped after a text change. */
    if (!deferred) {
      requestAnimationFrame(() => {
        if (lastGutter) renderGutter(lastGutter.marks, lastGutter.colour, true);
      });
    }
  }

  function renderShotBar() {
    const total = Math.max(0.001, (alignFrames(st.length) - 1) / FPS);
    const shots = parseShots(st.prompt || "", total);
    shotbar.replaceChildren();

    /* A thin ruler over the band. Without it the coloured strip reads as a
     * progress bar or a palette; one row of ticks and it is obviously time.
     * Minor ticks every second, a longer one with a label every five. */
    const ruler = el("div", "ruler");
    (() => {
      const minor = total > 30 ? 5 : 1;
      const major = total > 30 ? 15 : 5;
      for (let t = 0; t <= total + 1e-6; t += minor) {
        const x = (t / total) * 100;
        if (x > 100) break;
        const isMaj = Math.abs(t / major - Math.round(t / major)) < 1e-6;
        const tick = el("div", "t" + (isMaj ? " maj" : ""));
        tick.style.left = `${x}%`;
        tick.style.height = isMaj ? "6px" : "3px";
        ruler.append(tick);
        /* skip a label that would hang off the right edge */
        if (isMaj && x < 88) {
          const lb = el("div", "lbl", `${Math.round(t)}s`);
          lb.style.left = `${x}%`;
          ruler.append(lb);
        }
      }
    })();
    shotbar.append(ruler);

    const band = el("div", "band");
    const notes = [];

    if (!shots.length) {
      /* No markers is not an error -- one continuous shot is his default.
       * The band then just reads as the clip's length. */
      const seg = el("div", "seg");
      seg.style.flex = "1 1 0%";
      seg.style.background = SHOT_COLOURS[0] + "94";
      seg.title = `one continuous shot \u00b7 ${fmtSecs(total)}`;
      const only = (fieldText(st.prompt || "", "detailed_description")
                 || String(st.prompt || "")).replace(/\s+/g, " ").trim();
      if (only) seg.append(el("span", "lab", only));
      band.append(seg);
      renderGutter([], () => SHOT_COLOURS[0] + "94");
    } else {
      /* Segments are laid out from a live copy of the times so a boundary drag
       * can relayout without touching the prompt on every pointermove. The
       * text is rewritten once, on release. */
      const live = shots.filter((s) => !s.untimed && !s.past);
      shots.forEach((s) => {
        if (s.untimed) notes.push({ warn: true, text: `shot ${s.n} has no timestamp \u2014 it cannot be placed` });
        else if (s.past) notes.push({ warn: true, text: `shot ${s.n} starts at ${fmtSecs(s.start)}, after the clip ends` });
      });

      const times = live.map((s) => s.start);
      const segs = live.map((s, i) => {
        const seg = el("div", "seg" + (s.end > total ? " past" : ""));
        seg.style.background = SHOT_COLOURS[i % SHOT_COLOURS.length] + "b4";
        const blurb = shotBlurb(st.prompt, s);
        seg.append(el("span", "lab", blurb || `shot ${s.n}`));
        band.append(seg);
        return seg;
      });

      const bounds = [];
      const layout = () => {
        segs.forEach((seg, i) => {
          const from = Math.min(times[i], total);
          const to = Math.min(i + 1 < times.length ? times[i + 1] : total, total);
          seg.style.flex = `${Math.max(to - from, 0.02)} 0 0%`;
          const bl = shotBlurb(st.prompt, live[i]);
          seg.title = `shot ${live[i].n} \u00b7 ${fmtSecs(from)} \u2192 ${fmtSecs(to)} (${fmtSecs(to - from)})`
                    + (bl ? `\n${bl.slice(0, 300)}` : "");
        });
        bounds.forEach((h, k) => { h.style.left = `${(times[k + 1] / total) * 100}%`; });
      };

      /* Drag a boundary to move where a shot starts, and the timestamp in the
       * prompt is rewritten to match. Snaps to a tenth of a second, or to the
       * frame with Shift held. Boundaries cannot cross each other. */
      live.slice(1).forEach((s, k) => {
        const h = el("div", "bnd");
        h.title = "Drag to move where this shot starts";
        band.append(h);
        bounds.push(h);

        const timeAt = (clientX) => {
          const r = band.getBoundingClientRect();
          return Math.min(total, Math.max(0, ((clientX - r.left) / r.width) * total));
        };
        let dragging = false;
        const move = (e) => {
          if (!dragging) return;
          e.preventDefault(); e.stopPropagation();
          const raw = timeAt(e.clientX);
          const t = e.shiftKey ? Math.round(raw * FPS) / FPS : Math.round(raw * 10) / 10;
          const lo = times[k] + 0.2;
          const hi = (k + 2 < times.length ? times[k + 2] : total) - 0.2;
          times[k + 1] = Math.min(Math.max(t, lo), Math.max(lo, hi));
          layout();
        };
        const up = (e) => {
          if (!dragging) return;
          dragging = false;
          h.classList.remove("on");
          document.removeEventListener("pointermove", move, true);
          document.removeEventListener("pointerup", up, true);
          writeShotTime(s, times[k + 1]);
        };
        h.addEventListener("pointerdown", (e) => {
          e.preventDefault(); e.stopPropagation();
          dragging = true;
          h.classList.add("on");
          try { h.setPointerCapture(e.pointerId); } catch (_) {}
          document.addEventListener("pointermove", move, true);
          document.addEventListener("pointerup", up, true);
        });
      });

      layout();
      renderGutter(live, (i) => SHOT_COLOURS[i % SHOT_COLOURS.length] + "b4");

      /* A gap before the first marker is real dead air, and H3 fills silence
       * with invented sound -- so it is worth seeing, not hiding. */
      const first = shots.find((s) => !s.untimed && !s.past);
      if (first && first.start > 0.01) {
        const lead = el("div", "seg");
        lead.style.flex = `${first.start} 0 0%`;
        lead.style.background = "#ffffff14";
        lead.title = `nothing scripted before ${fmtSecs(first.start)}`;
        band.prepend(lead);
        notes.push({ warn: false, text: `${fmtSecs(first.start)} before the first shot` });
      }
      const timed = shots.filter((s) => !s.untimed && !s.past);
      const last = timed[timed.length - 1];
      if (last && last.end > total + 0.01) {
        notes.push({ warn: true, text: `shot ${last.n} runs past the clip by ${fmtSecs(last.end - total)}` });
      }
    }

    shotbar.append(band);
    if (notes.length) {
      shotbar.append(el("div", "note" + (notes.some((n) => n.warn) ? " warn" : ""),
        notes.map((n) => n.text).join("  \u00b7  ")));
    }
  }

  function renderTags() {
    const { rows, tags } = presentation(st);

    /* Rebuild the chips ONLY when the reference set changes. Typing calls
     * this on every keystroke, and recreating a <video> thumbnail makes it
     * reload -- which is the flicker. */
    const sig = rows.map((r) => (tokenOf(r) || "") + r.tag).join("|");
    if (sig !== chipSig) {
      chipSig = sig;
      chipEls.clear();
      chips.replaceChildren();
      rows.forEach((r) => {
        const token = tokenOf(r);
        if (!token) return;
        const b = el("button", "gcast-chip");
        const slotFile = fileForToken(token);
        if (r.kind === "image" && slotFile) {
          const im = el("img"); im.src = viewURL(slotFile); b.append(im);
        } else if (r.kind === "video" && slotFile) {
          const v = el("video");
          v.src = viewURL(slotFile); v.muted = true; v.preload = "metadata"; v.playsInline = true;
          b.append(v);
        } else {
          b.append(el("span", "glyph", "♪"));
        }
        b.append(el("span", null, token));
        b.onclick = () => insert(token);
        b.dataset.tag = r.tag;
        chips.append(b);
        chipEls.set(token, b);
      });
      if (!rows.length) chips.append(el("span", "gcast-read", "No references yet — fill a slot to get a tag."));
    }
    /* the in-the-prompt state changes as you type, so only that is repainted */
    chipEls.forEach((b, token) => {
      const on = hasToken(token);
      b.classList.toggle("on", on);
      b.title = on
        ? `${token} → ${b.dataset.tag} · in the prompt`
        : `insert ${token} → ${b.dataset.tag}`;
    });

    /* presentation strip */
    pres.replaceChildren();
    if (!rows.length) {
      pres.append(el("span", null, "prompt only"));
    } else {
      let silent = 0;
      rows.forEach((r) => {
        const item = el("span");
        const tok = r.token || (r.from === "first frame" ? "@first" : r.from === "last frame" ? "@last" : null);
        const used = hasToken(tok);
        if (!used) silent++;
        item.innerHTML = `<b>${r.tag.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</b> <span class="arrow">←</span> ${r.from}`;
        if (!used) item.classList.add("gcast-warn");
        pres.append(item);
      });
      if (silent) pres.append(el("span", "gcast-warn", `· ${silent} never mentioned in the prompt`));
    }
    const orphans = ALL_TOKENS.filter((t) => !tags[t] && hasToken(t));
    if (orphans.length) {
      pres.append(el("span", "gcast-warn", `· ${orphans.join(" ")} points at an empty slot`));
      pres.append(el("span", "arrow", "then the prompt"));
    }
  }

  /* ---- preset files ------------------------------------------------
   *
   * Everything happens in the browser: a .json holds the state and points
   * at media by filename, a .h3pack is a zip with the media inside. The
   * media itself is read back out of ComfyUI through /view and pushed
   * back in through /upload on import, so no server route is needed.
   *
   * Reference images are re-encoded to JPEG inside a pack -- they are
   * heading into a VAE, so the artifacts are irrelevant and the saving is
   * large. First/last KEYFRAMES are deliberately left alone: those are the
   * chunk-continuity frames, and JPEG damage there propagates into every
   * generation built on top of them.
   */

  const JPEG_QUALITY = 0.92;

  let fileHandle = null;      // File System Access handle, when supported
  let fileIsPack = false;
  let fileLabel = "";

  /* -- minimal zip, stored (media is already compressed) -- */
  const CRC_TABLE = (() => {
    const t = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c >>> 0;
    }
    return t;
  })();

  function crc32(buf) {
    let c = 0xFFFFFFFF;
    for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
    return (c ^ 0xFFFFFFFF) >>> 0;
  }

  function zipWrite(entries) {
    const enc = new TextEncoder();
    const parts = [], central = [];
    let offset = 0;

    for (const e of entries) {
      const name = enc.encode(e.name);
      const data = e.data;
      const crc = crc32(data);

      const lh = new DataView(new ArrayBuffer(30));
      lh.setUint32(0, 0x04034b50, true);
      lh.setUint16(4, 20, true);        // version needed
      lh.setUint16(8, 0, true);         // stored
      lh.setUint32(14, crc, true);
      lh.setUint32(18, data.length, true);
      lh.setUint32(22, data.length, true);
      lh.setUint16(26, name.length, true);
      parts.push(new Uint8Array(lh.buffer), name, data);

      const cd = new DataView(new ArrayBuffer(46));
      cd.setUint32(0, 0x02014b50, true);
      cd.setUint16(4, 20, true);
      cd.setUint16(6, 20, true);
      cd.setUint16(10, 0, true);
      cd.setUint32(16, crc, true);
      cd.setUint32(20, data.length, true);
      cd.setUint32(24, data.length, true);
      cd.setUint16(28, name.length, true);
      cd.setUint32(42, offset, true);
      central.push(new Uint8Array(cd.buffer), name);

      offset += 30 + name.length + data.length;
    }

    let cdSize = 0;
    for (const c of central) cdSize += c.length;
    const eo = new DataView(new ArrayBuffer(22));
    eo.setUint32(0, 0x06054b50, true);
    eo.setUint16(8, entries.length, true);
    eo.setUint16(10, entries.length, true);
    eo.setUint32(12, cdSize, true);
    eo.setUint32(16, offset, true);

    return new Blob([...parts, ...central, new Uint8Array(eo.buffer)],
                    { type: "application/zip" });
  }

  async function zipRead(buffer) {
    const dv = new DataView(buffer);
    const bytes = new Uint8Array(buffer);
    let eocd = -1;
    for (let i = dv.byteLength - 22; i >= 0 && i > dv.byteLength - 66000; i--) {
      if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error("not a zip file");

    const count = dv.getUint16(eocd + 10, true);
    let p = dv.getUint32(eocd + 16, true);
    const dec = new TextDecoder();
    const out = new Map();

    for (let i = 0; i < count; i++) {
      const nameLen = dv.getUint16(p + 28, true);
      const extraLen = dv.getUint16(p + 30, true);
      const commentLen = dv.getUint16(p + 32, true);
      const method = dv.getUint16(p + 10, true);
      const compSize = dv.getUint32(p + 20, true);
      const local = dv.getUint32(p + 42, true);
      const name = dec.decode(bytes.subarray(p + 46, p + 46 + nameLen));

      const lNameLen = dv.getUint16(local + 26, true);
      const lExtraLen = dv.getUint16(local + 28, true);
      const start = local + 30 + lNameLen + lExtraLen;
      let data = bytes.subarray(start, start + compSize);

      if (method === 8) {
        const ds = new DecompressionStream("deflate-raw");
        const blob = new Blob([data]).stream().pipeThrough(ds);
        data = new Uint8Array(await new Response(blob).arrayBuffer());
      } else if (method !== 0) {
        throw new Error("unsupported zip compression in " + name);
      }
      out.set(name, data);
      p += 46 + nameLen + extraLen + commentLen;
    }
    return out;
  }

  /* -- media helpers -- */

  async function toJpeg(blob) {
    try {
      const bmp = await createImageBitmap(blob);
      const c = document.createElement("canvas");
      c.width = bmp.width; c.height = bmp.height;
      c.getContext("2d").drawImage(bmp, 0, 0);
      const out = await new Promise((r) => c.toBlob(r, "image/jpeg", JPEG_QUALITY));
      bmp.close?.();
      return out;
    } catch (e) { return null; }
  }

  const bytesOf = async (blob) => new Uint8Array(await blob.arrayBuffer());

  function busy(text) {
    [bSave, bSaveAs, bPack, bLoad].forEach((b) => { b.disabled = !!text; b.style.opacity = text ? .5 : 1; });
    if (text) nameLabel.innerHTML = `<span class="dirty">${text}</span>`;
    else paintPresetName();
  }

  /* -- build a pack: state plus every referenced file -- */
  /* Collects media into zip entries, rewriting each item's filename to point
   * at the copy inside the pack. The dedup map is passed in rather than owned,
   * so a whole project can share ONE map across every shot: a reference sheet
   * used in ten shots is fetched once and stored once. */
  function makeAssetAdder(entries, seen, used) {
    return async function add(item, keepOriginal) {
      if (!item || !item.file) return;
      const src = item.file;
      if (seen.has(src)) { item.file = seen.get(src); return; }

      let blob;
      try { blob = await (await fetch(viewURL(src))).blob(); }
      catch (e) { console.warn("[H3 Studio] could not read " + src, e); return; }

      let name = src.split("/").pop();
      if (!keepOriginal && /\.(png|jpe?g|webp|bmp)$/i.test(name)) {
        const j = await toJpeg(blob);
        if (j && j.size < blob.size) {
          blob = j;
          name = name.replace(/\.[^.]+$/, "") + ".jpg";
        }
      }
      let unique = name, n = 2;
      while (used.has(unique)) {
        unique = name.replace(/(\.[^.]+)$/, `_${n++}$1`);
      }
      used.add(unique);
      entries.push({ name: "assets/" + unique, data: await bytesOf(blob) });
      seen.set(src, unique);
      item.file = unique;
    };
  }

  async function packSlots(slots, add) {
    if (!slots) return;
    await add(slots.first, true);                 // keyframes stay lossless
    await add(slots.last, true);
    for (const it of slots.images || []) await add(it, false);
    for (const it of slots.videos || []) await add(it, false);
    for (const it of slots.audios || []) await add(it, false);
  }

  async function buildPack() {
    const state = JSON.parse(JSON.stringify(st));
    const entries = [];
    const add = makeAssetAdder(entries, new Map(), new Set());

    await packSlots(state.slots, add);

    const enc = new TextEncoder();
    entries.unshift({ name: "state.json", data: enc.encode(JSON.stringify(state, null, 1)) });
    entries.unshift({ name: "meta.json", data: enc.encode(JSON.stringify({
      app: "H3 Studio", version: 1,
      mode: state.mode, width: state.width, height: state.height,
      length: state.length, saved: new Date().toISOString(),
    }, null, 1)) });

    return zipWrite(entries);
  }

  function plainBlob() {
    return new Blob([JSON.stringify({
      meta: { app: "H3 Studio", version: 1, mode: st.mode, packed: false,
              saved: new Date().toISOString() },
      state: st,
    }, null, 1)], { type: "application/json" });
  }

  const defaultName = (pack) =>
    `${st.mode === "fl2va" ? "h3-fl" : "h3-ref"}-${st.width}x${st.height}` +
    (pack ? ".h3pack" : ".h3.json");

  /* -- OS dialogs, with a download fallback where the API is missing -- */

  async function writeOut(blob, suggested, pack) {
    if (window.showSaveFilePicker) {
      try {
        const handle = await window.showSaveFilePicker({
          suggestedName: suggested,
          types: [pack
            ? { description: "H3 Studio pack", accept: { "application/zip": [".h3pack"] } }
            : { description: "H3 Studio preset", accept: { "application/json": [".json"] } }],
        });
        const w = await handle.createWritable();
        await w.write(blob);
        await w.close();
        fileHandle = handle; fileIsPack = pack; fileLabel = handle.name;
        return true;
      } catch (e) {
        if (e && e.name === "AbortError") return false;
        console.warn("[H3 Studio] save picker failed, downloading instead", e);
      }
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = suggested;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    fileHandle = null; fileIsPack = pack; fileLabel = suggested;
    return true;
  }

  async function saveAs(pack) {
    busy(pack ? "packing\u2026" : "saving\u2026");
    try {
      const blob = pack ? await buildPack() : plainBlob();
      await writeOut(blob, defaultName(pack), pack);
    } catch (e) {
      alert("H3 Studio: could not save \u2014 " + e);
    } finally { busy(null); }
  }

  async function saveOver() {
    if (!fileHandle) return saveAs(fileIsPack);
    busy(fileIsPack ? "packing\u2026" : "saving\u2026");
    try {
      const blob = fileIsPack ? await buildPack() : plainBlob();
      const w = await fileHandle.createWritable();
      await w.write(blob);
      await w.close();
    } catch (e) {
      alert("H3 Studio: could not save \u2014 " + e);
    } finally { busy(null); }
  }

  async function pickFileForOpen() {
    if (window.showOpenFilePicker) {
      try {
        const [h] = await window.showOpenFilePicker({
          types: [{ description: "H3 Studio preset or pack",
                    accept: { "application/json": [".json"], "application/zip": [".h3pack", ".h3projpack", ".zip"] } }],
        });
        const f = await h.getFile();
        return { file: f, handle: h };
      } catch (e) {
        if (e && e.name === "AbortError") return null;
      }
    }
    const f = await pickFile(".json,.h3pack,.h3projpack,.zip");
    return f ? { file: f, handle: null } : null;
  }

  /* Read a .h3.json or .h3pack into a plain state object. Split out of
   * doLoad so importing a saved shot into a project takes the identical
   * path, pack media restore included. */
  async function stateFromFile(file) {
    const isPack = /\.(h3pack|zip)$/i.test(file.name);
    if (!isPack) {
      const parsed = JSON.parse(await file.text());
      return { state: parsed.state || parsed, isPack: false };
    }
    const map = await zipRead(await file.arrayBuffer());
    const raw = map.get("state.json");
    if (!raw) throw new Error("pack has no state.json");
    const state = JSON.parse(new TextDecoder().decode(raw));

    /* push the bundled media back into ComfyUI's input folder so the
     * Python side can find it, then repoint the state at the uploads */
    const remap = new Map();
    for (const [name, data] of map) {
      if (!name.startsWith("assets/")) continue;
      const base = name.slice(7);
      const blob = new Blob([data]);
      try {
        remap.set(base, await uploadFile(new File([blob], base)));
      } catch (e) {
        console.warn("[H3 Studio] could not restore " + base, e);
      }
    }
    const fix = (item) => {
      if (item && item.file && remap.has(item.file)) item.file = remap.get(item.file);
    };
    const sl = state.slots || {};
    fix(sl.first); fix(sl.last);
    (sl.images || []).forEach(fix);
    (sl.videos || []).forEach(fix);
    (sl.audios || []).forEach(fix);
    return { state, isPack: true };
  }

  /* Same dialog, several files at once - importing a finished film one shot
   * at a time was the tedious part. */
  async function pickFilesForOpen() {
    if (window.showOpenFilePicker) {
      try {
        const hs = await window.showOpenFilePicker({
          multiple: true,
          types: [{ description: "H3 Studio preset or pack",
                    accept: { "application/json": [".json"], "application/zip": [".h3pack", ".h3projpack", ".zip"] } }],
        });
        const out = [];
        for (const h of hs) out.push(await h.getFile());
        return out;
      } catch (e) {
        if (e && e.name === "AbortError") return [];
      }
    }
    return (await pickFile(".json,.h3pack,.h3projpack,.zip", true)) || [];
  }

  async function doLoad() {
    const picked = await pickFileForOpen();
    if (!picked) return;
    const { file, handle } = picked;
    busy("loading\u2026");
    try {
      const { state, isPack } = await stateFromFile(file);
      load(JSON.stringify(state));
      fileHandle = handle; fileIsPack = isPack; fileLabel = file.name;
      commit();
    } catch (e) {
      alert("H3 Studio: could not load \u2014 " + e);
    } finally { busy(null); }
  }

  function paintPresetName() {
    const isFl = st.mode === "fl2va";
    const badge = `<span class="gcast-badge ${isFl ? "fl" : "ref"}">${isFl ? "FL" : "REF"}</span>`;
    /* Save / Save as / Save packed always write THIS SHOT, never the project.
     * With a project open that is easy to forget, so the bar names the shot
     * those buttons would write. */
    const p = (node.properties && node.properties.gcast_project) || null;
    const shot = (p && Array.isArray(p.shots) && p.idx >= 0 && p.shots[p.idx])
      ? `<span class="shot">clip ${p.idx + 1}/${p.shots.length}</span>` : "";
    nameLabel.innerHTML = (fileLabel
      ? `${badge} ${fileLabel}`
      : `${badge} <span class="dirty">unsaved</span>`) + (shot ? " " + shot : "");
  }

  /* ---- shots: several shots in one project --------------------------
   *
   * A film is a list of shots that mostly share their references, so a file
   * per shot loses the relationship and repeats the work. A project holds
   * the shot states together.
   *
   * The single-shot Save / Save as / Save packed / Load buttons are NOT
   * touched: every existing .h3.json and .h3pack keeps loading exactly as
   * before, and a project is a separate file with its own two buttons.
   *
   * Shots live in node.properties, which LiteGraph serialises with the
   * workflow. A project therefore survives a browser reload without ever
   * being written to disk -- the file is for moving it somewhere else, not
   * for not losing it.
   *
   * Media is referenced by filename, as in .h3.json. There is no packed
   * project yet: the files have to still be in ComfyUI's input folder.
   */

  const clone = (o) => JSON.parse(JSON.stringify(o));
  const uid = () => Math.random().toString(36).slice(2, 9);
  let projHandle = null;
  let projLabel = "";
  let shotsOpen = false;
  let shotsFocus = -1;

  function proj() {
    node.properties = node.properties || {};
    const p = node.properties.gcast_project;
    if (!p || !Array.isArray(p.shots)) {
      node.properties.gcast_project = { name: "", shots: [], idx: -1 };
    }
    return node.properties.gcast_project;
  }

  /* ---- previous state: the .bak, kept in the workflow ----------------
   *
   * There is exactly ONE copy of a project, in node.properties. New, Open,
   * Import and Delete each rewrite that copy, and the file on disk is only
   * as recent as the last time Save was pressed. So before anything that
   * rewrites the list, the current one is put aside in a second property
   * and the panel's Revert button comes alive.
   *
   * Kept in properties rather than written next to the project file on
   * purpose: a file handle can only rewrite the file it points at, so a
   * sibling .bak needs a DIRECTORY handle - and where the File System
   * Access API is missing there is no dialog at all and saves land in the
   * downloads folder. Properties work everywhere and survive a reload.
   *
   * ONE level deep: this is a .bak, not an undo stack. A backup of a
   * backup is not worth the workflow bytes.
   *
   * Deliberately NOT taken on the autosave that fires when you switch
   * clip. That runs constantly while browsing, so the stored state would
   * almost always be "the same project, one clip ago" - useless as a
   * safety net, and it would flush out a snapshot worth having.
   */
  function projSnapshot() {
    const p = node.properties && node.properties.gcast_project;
    if (!p || !Array.isArray(p.shots) || !p.shots.length) return null;
    return { name: p.name || "", idx: p.idx, shots: clone(p.shots) };
  }

  function snapProject(why) {
    const snap = projSnapshot();
    if (!snap) return;                     // an empty project has nothing to lose
    node.properties.gcast_project_prev = {
      why: why || "", at: Date.now(),
      name: snap.name, idx: snap.idx, shots: snap.shots,
    };
  }

  function prevProject() {
    const b = node.properties && node.properties.gcast_project_prev;
    return (b && Array.isArray(b.shots) && b.shots.length) ? b : null;
  }

  function agoText(ms) {
    const s = Math.max(0, Math.round((Date.now() - ms) / 1000));
    if (s < 60) return "a moment ago";
    const m = Math.round(s / 60);
    if (m < 60) return m + " min ago";
    const h = Math.round(m / 60);
    return h < 24 ? h + " h ago" : Math.round(h / 24) + " d ago";
  }

  function prevWhen(b) {
    return agoText(b.at) + (b.why ? " (" + b.why + ")" : "");
  }

  /* Revert SWAPS the two, so a mis-click on Revert is itself revertible. */
  function revertProject() {
    const b = prevProject();
    if (!b) return;
    const n = b.shots.length;
    if (!confirm(
          `Go back to the project as it was ${prevWhen(b)}?\n\n` +
          `${n} clip${n > 1 ? "s" : ""}. The project you have now is kept as the previous ` +
          `state, so pressing Revert again brings it back.`)) return;
    stash();
    const now = projSnapshot();
    node.properties.gcast_project = { name: b.name || "", idx: b.idx, shots: clone(b.shots) };
    if (now) {
      node.properties.gcast_project_prev = {
        why: "before Revert", at: Date.now(),
        name: now.name, idx: now.idx, shots: now.shots,
      };
    } else {
      delete node.properties.gcast_project_prev;
    }
    const p = proj();
    p.idx = p.shots.length ? Math.min(Math.max(0, +p.idx || 0), p.shots.length - 1) : -1;
    if (p.idx >= 0) load(JSON.stringify(p.shots[p.idx].state));
    commit(); paintShotsBtn(); paintPresetName(); renderShots();
  }

  /* Autosave. The shot being left is written back before anything else
   * happens, so browsing the list can never cost an edit. */
  function stash() {
    const p = proj();
    if (p.idx >= 0 && p.shots[p.idx]) p.shots[p.idx].state = clone(st);
  }

  /* Vocabulary, kept straight on purpose: a CLIP is one generation, and a clip
   * can contain several SHOTS - the [Shot N] markers the model cuts on. The
   * project list talks about clips; the timeline strip talks about shots.
   * Internals and the file format still say "shots" so old projects load. */
  const shotLabel = (s, i) => s.name || `Clip ${i + 1}`;

  function shotThumb(state) {
    const sl = (state && state.slots) || {};
    const cand = [sl.first, sl.last].concat(sl.images || [], sl.videos || []);
    const hit = cand.find((x) => x && x.file);
    return hit ? hit.file : null;
  }

  function paintShotsBtn() {
    const p = proj();
    bShots.textContent = p.shots.length
      ? `Project ${p.idx >= 0 ? p.idx + 1 : "-"}/${p.shots.length}`
      : "Project";
  }

  /* A new shot in a film usually reuses the same cast and location, so it
   * starts from the current one. The prompt does not carry over -- except
   * subject_definitions, which describes the references that just came with
   * it and would only be retyped. */
  function inheritedState() {
    const s = clone(st);
    const subs = fieldText(st.prompt || "", "subject_definitions");
    s.prompt = subs ? `subject_definitions:\n${subs}\n\n` : "";
    return s;
  }

  /* The first Add turns what is on screen into Shot 1 rather than losing it. */
  function seedFromScreen(p) {
    if (p.shots.length === 0 && p.idx < 0) {
      p.shots.push({ id: uid(), name: "", state: clone(st) });
      p.idx = 0;
    }
  }

  function addShot(blank) {
    const p = proj();
    stash();
    seedFromScreen(p);
    const s = blank ? blankState() : inheritedState();
    p.shots.push({ id: uid(), name: "", state: s });
    p.idx = p.shots.length - 1;
    load(JSON.stringify(s));
    commit(); paintShotsBtn(); paintPresetName(); renderShots();
  }

  function switchTo(i) {
    const p = proj();
    if (i === p.idx || !p.shots[i]) return;
    stash();
    p.idx = i;
    load(JSON.stringify(p.shots[i].state));
    commit(); paintShotsBtn(); paintPresetName(); shotsFocus = i; renderShots();
  }

  function delShot(i) {
    const p = proj();
    const s = p.shots[i];
    if (!s) return;
    if (!confirm(`Delete ${shotLabel(s, i)}?`)) return;
    snapProject("before deleting " + shotLabel(s, i));
    p.shots.splice(i, 1);
    if (!p.shots.length) p.idx = -1;
    else if (p.idx > i) p.idx--;
    else if (p.idx === i) {
      p.idx = Math.min(i, p.shots.length - 1);
      load(JSON.stringify(p.shots[p.idx].state));
    }
    commit(); paintShotsBtn(); paintPresetName(); renderShots();
  }

  function moveShot(i, d) {
    const p = proj();
    const j = i + d;
    if (j < 0 || j >= p.shots.length) return;
    const [it] = p.shots.splice(i, 1);
    p.shots.splice(j, 0, it);
    if (p.idx === i) p.idx = j;
    else if (p.idx === j) p.idx = i;
    shotsFocus = j;
    commit(); paintShotsBtn(); paintPresetName(); renderShots();
  }

  async function importShot() {
    const files = await pickFilesForOpen();
    if (!files.length) return;
    busy(files.length > 1 ? `importing ${files.length}\u2026` : "importing\u2026");
    const failed = [];
    try {
      const p = proj();
      stash();
      snapProject("before Import");
      seedFromScreen(p);
      /* name order, so a folder of shot-01 … shot-12 lands in film order */
      files.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
      for (const f of files) {
        try {
          const { state } = await stateFromFile(f);
          p.shots.push({
            id: uid(),
            name: f.name.replace(/\.(h3\.json|h3pack|json|zip)$/i, ""),
            state: parseInitial(JSON.stringify(state)),
          });
        } catch (e) {
          console.warn("[H3 Studio] could not import " + f.name, e);
          failed.push(f.name);
        }
      }
      p.idx = p.shots.length - 1;
      load(JSON.stringify(p.shots[p.idx].state));
      commit(); paintShotsBtn(); paintPresetName(); shotsFocus = p.idx; renderShots();
      if (failed.length) alert("H3 Studio: could not import \u2014 " + failed.join(", "));
    } catch (e) {
      alert("H3 Studio: could not import \u2014 " + e);
    } finally { busy(null); }
  }

  function projectBlob() {
    const p = proj();
    return new Blob([JSON.stringify({
      meta: { app: "H3 Studio project", version: 1, saved: new Date().toISOString() },
      name: p.name || "", idx: p.idx, shots: p.shots,
    }, null, 1)], { type: "application/json" });
  }

  async function saveProject(asNew) {
    const p = proj();
    stash();
    if (!p.shots.length) { alert("H3 Studio: this project has no clips yet."); return; }
    if (asNew) projHandle = null;
    busy("saving project\u2026");
    let wrote = false;
    try {
      const blob = projectBlob();
      const stem = (p.name || "h3-project").replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "-") || "h3-project";
      const suggested = stem + ".h3proj.json";
      if (projHandle) {
        const w = await projHandle.createWritable();
        await w.write(blob); await w.close();
        projLabel = projHandle.name;
        wrote = true;
      } else if (window.showSaveFilePicker) {
        try {
          const h = await window.showSaveFilePicker({
            suggestedName: suggested,
            types: [{ description: "H3 Studio project", accept: { "application/json": [".json"] } }],
          });
          const w = await h.createWritable();
          await w.write(blob); await w.close();
          projHandle = h; projLabel = h.name;
          wrote = true;
        } catch (e) { if (!(e && e.name === "AbortError")) throw e; }
      } else {
        /* No File System Access here, so the file lands in the browser's
         * download folder with no dialog at all. Say so - a silent save
         * reads as a save that did not happen. */
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = suggested;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 10000);
        projLabel = suggested;
        wrote = true;
        alert("H3 Studio: your browser has no save dialog here, so the project was downloaded as \u201c"
              + suggested + "\u201d to your usual downloads folder.");
      }
      /* Snapshot AFTER a write that actually happened, never after an
       * abandoned dialog. Revert then means "back to what I last saved",
       * which is the useful reading whenever nothing destructive has
       * happened since - and if something has, that action took its own
       * snapshot and this one is already gone. */
      if (wrote) snapProject("as last saved");
      renderShots();
    } catch (e) {
      alert("H3 Studio: could not save the project \u2014 " + e);
    } finally { busy(null); }
  }

  /* Close the project without touching the shot on screen. Clearing the list
   * and wiping the node are different wishes: the usual reason to start a new
   * project is that the shot you are looking at is the first one of it, so it
   * is kept and the next Add makes it Shot 1. "Blank" still gives an empty
   * node when that is what you want. */
  function newProject() {
    const p = proj();
    if (p.shots.length) {
      const n = p.shots.length;
      const ok = confirm(
        `Close ${p.name || "this project"} and its ${n} clip${n > 1 ? "s" : ""}?\n\n` +
        "The clip on screen is kept. Anything you have not written to a project file is gone.");
      if (!ok) return;
    }
    /* The list goes, the copy stays: Revert is the way back. */
    snapProject("before New");
    node.properties.gcast_project = { name: "", shots: [], idx: -1 };
    projHandle = null; projLabel = "";
    commit(); paintShotsBtn(); paintPresetName(); renderShots();
  }

  /* A packed project: one zip, every shot, media inside, dedup shared across
   * shots. The .h3proj.json stays the working file; this is the deliverable.
   * So packing always asks where to put it and never becomes projHandle -
   * otherwise "Save project" would start silently rewriting a huge export. */
  async function packProject() {
    const p = proj();
    stash();
    if (!p.shots.length) { alert("H3 Studio: this project has no clips yet."); return; }
    busy("packing project\u2026");
    try {
      const shots = clone(p.shots);
      const entries = [];
      const add = makeAssetAdder(entries, new Map(), new Set());
      for (const s of shots) await packSlots(s.state.slots, add);

      const enc = new TextEncoder();
      entries.unshift({ name: "project.json", data: enc.encode(JSON.stringify({
        meta: { app: "H3 Studio project", version: 1, packed: true,
                shots: shots.length, saved: new Date().toISOString() },
        name: p.name || "", idx: p.idx, shots,
      }, null, 1)) });

      const stem = (p.name || "h3-project").replace(/[^\w\- ]+/g, "").trim().replace(/\s+/g, "-") || "h3-project";
      /* Named .h3proj.zip rather than .h3projpack: it IS a zip, and an
       * extension the OS recognises is one less thing to explain to whoever
       * you hand it to. openProject still accepts the old .h3projpack. */
      const suggested = stem + ".h3proj.zip";

      /* The zip writer uses 32-bit offsets and sizes (no ZIP64) and stores
       * entries uncompressed, so the whole archive has to stay under 4 GB.
       * Video reference clips are packed WHOLE - trim points are metadata -
       * so a few masters get there faster than you would think. Better to
       * say so now than to hand someone a silently corrupt file. */
      let total = 0;
      for (const e of entries) total += e.data.length;
      if (total > 3.8e9) {
        const gb = (total / 1e9).toFixed(1);
        alert(`H3 Studio: this project would pack to about ${gb} GB, past the 4 GB `
            + `limit of the pack format. Trim or re-encode the reference videos `
            + `first \u2014 they are stored whole, however short the trim window is.`);
        return;
      }
      const blob = await zipWrite(entries);

      if (window.showSaveFilePicker) {
        try {
          const h = await window.showSaveFilePicker({
            suggestedName: suggested,
            types: [{ description: "H3 Studio project pack",
                      accept: { "application/zip": [".zip", ".h3projpack"] } }],
          });
          const w = await h.createWritable();
          await w.write(blob); await w.close();
        } catch (e) { if (!(e && e.name === "AbortError")) throw e; }
      } else {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = suggested;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 20000);
        alert("H3 Studio: your browser has no save dialog here, so the pack was downloaded as \u201c"
              + suggested + "\u201d to your usual downloads folder.");
      }
    } catch (e) {
      alert("H3 Studio: could not pack the project \u2014 " + e);
    } finally { busy(null); }
  }

  async function openProject() {
    const picked = await pickFileForOpen();
    if (!picked) return;
    busy("loading project\u2026");
    try {
      const isPack = /\.(h3projpack|zip)$/i.test(picked.file.name);
      let d, remap = null;

      if (isPack) {
        const map = await zipRead(await picked.file.arrayBuffer());
        const raw = map.get("project.json");
        if (!raw) throw new Error("pack has no project.json");
        d = JSON.parse(new TextDecoder().decode(raw));
        /* restore each asset once, however many shots point at it */
        remap = new Map();
        for (const [name, data] of map) {
          if (!name.startsWith("assets/")) continue;
          const base = name.slice(7);
          try {
            remap.set(base, await uploadFile(new File([new Blob([data])], base)));
          } catch (e) {
            console.warn("[H3 Studio] could not restore " + base, e);
          }
        }
      } else {
        d = JSON.parse(await picked.file.text());
      }

      if (!d || !Array.isArray(d.shots)) throw new Error("not an H3 Studio project file");
      const p = proj();
      /* opening replaces whatever is in the list, same as closing it */
      if (p.shots.length && !confirm(
            `Replace the current project (${p.shots.length} clip${p.shots.length > 1 ? "s" : ""}) ` +
            `with ${picked.file.name}?`)) return;
      snapProject("before opening " + picked.file.name);
      p.name = typeof d.name === "string" ? d.name : "";
      /* every shot goes through parseInitial, so a project written by an
       * older build cannot smuggle in fields this one does not know */
      p.shots = d.shots.map((s) => ({
        id: (s && s.id) ? String(s.id) : uid(),
        name: (s && typeof s.name === "string") ? s.name : "",
        state: parseInitial(JSON.stringify((s && s.state) || {})),
      }));
      if (remap) {
        const fix = (it) => { if (it && it.file && remap.has(it.file)) it.file = remap.get(it.file); };
        p.shots.forEach((s) => {
          const sl = s.state.slots || {};
          fix(sl.first); fix(sl.last);
          (sl.images || []).forEach(fix);
          (sl.videos || []).forEach(fix);
          (sl.audios || []).forEach(fix);
        });
      }
      p.idx = p.shots.length ? Math.min(Math.max(0, +d.idx || 0), p.shots.length - 1) : -1;
      /* a pack is an import, not a working file - don't let Save write over it */
      projHandle = isPack ? null : (picked.handle || null);
      projLabel = picked.file.name;
      if (p.idx >= 0) load(JSON.stringify(p.shots[p.idx].state));
      commit(); paintShotsBtn(); paintPresetName(); renderShots();
    } catch (e) {
      alert("H3 Studio: could not open the project \u2014 " + e);
    } finally { busy(null); }
  }

  /* ---- the panel ---- */

  const shotsPanel = el("div", "gcast-shots");
  shotsPanel.style.display = "none";
  document.body.append(shotsPanel);   // outside the node: a popup inside it reflows the layout

  function placeShots() {
    const r = bShots.getBoundingClientRect();
    const w = 336;
    shotsPanel.style.left = Math.max(8, Math.min(window.innerWidth - w - 8, r.right - w)) + "px";
    shotsPanel.style.top = (r.bottom + 6) + "px";
  }

  function outsideShots(e) {
    if (shotsPanel.contains(e.target) || bShots.contains(e.target)) return;
    closeShots();
  }

  function openShots() {
    shotsOpen = true;
    shotsPanel.style.display = "";
    placeShots(); renderShots();
    /* Dismiss on pointerUP, never pointerdown: taking the panel off the page
     * under a press that is still down leaves the v2 canvas mid-drag. */
    setTimeout(() => document.addEventListener("pointerup", outsideShots, true), 0);
  }

  function closeShots() {
    shotsOpen = false;
    shotsPanel.style.display = "none";
    document.removeEventListener("pointerup", outsideShots, true);
  }

  function renderShots() {
    if (!shotsOpen) return;
    const p = proj();
    shotsPanel.dataset.mode = st.mode;

    /* The list is rebuilt on every change, which used to throw the scroll
     * back to the top - so moving a shot down twice meant scrolling down
     * twice. Keep the offset, and pull the row that just moved into view. */
    const prevList = shotsPanel.querySelector(".gcast-shots-list");
    const prevTop = prevList ? prevList.scrollTop : 0;
    shotsPanel.textContent = "";

    const head = el("div", "gcast-shots-head");
    head.append(el("div", "gcast-shots-title", "Project"));
    const nm = el("input");
    nm.type = "text";
    nm.placeholder = "Project name";
    nm.value = p.name || "";
    nm.oninput = () => { p.name = nm.value; commit(); };
    nm.onkeydown = (e) => e.stopPropagation();
    head.append(nm);
    head.append(el("div", "gcast-shots-file",
      projLabel ? `file: ${projLabel}` : "not written to a file yet \u2014 kept in the workflow"));
    shotsPanel.append(head);

    const list = el("div", "gcast-shots-list");
    if (!p.shots.length) {
      list.append(el("div", "gcast-shots-empty",
        "No clips yet. \u201cAdd clip\u201d keeps what is on screen as Clip 1 and starts Clip 2 from its references."));
    }
    p.shots.forEach((s, i) => {
      const row = el("div", "gcast-shot" + (i === p.idx ? " on" : ""));

      const th = el("div", "th");
      const f = shotThumb(s.state);
      if (f) {
        const isVid = /\.(mp4|webm|mov|mkv|m4v|avi)$/i.test(f);
        const m = el(isVid ? "video" : "img");
        m.src = viewURL(f);
        if (isVid) { m.muted = true; m.preload = "metadata"; }
        th.append(m);
      } else th.append(el("span", null, "\u2014"));

      const mid = el("div", "mid");
      const nameEl = el("div", "nm", shotLabel(s, i));
      nameEl.title = "Click to rename";
      nameEl.onclick = (e) => {
        e.stopPropagation();
        const inp = el("input");
        inp.type = "text";
        inp.value = s.name || "";
        inp.placeholder = `Clip ${i + 1}`;
        nameEl.replaceWith(inp);
        inp.focus(); inp.select();
        const done = () => { s.name = inp.value.trim(); commit(); renderShots(); };
        inp.onclick = (ev) => ev.stopPropagation();
        inp.onblur = done;
        inp.onkeydown = (ev) => {
          ev.stopPropagation();
          if (ev.key === "Enter") { ev.preventDefault(); done(); }
          if (ev.key === "Escape") { ev.preventDefault(); renderShots(); }
        };
      };
      const dur = fmtSecs(Math.max(0, ((s.state.length || 5) - 1) / FPS));
      mid.append(nameEl, el("div", "meta",
        `${s.state.mode === "fl2va" ? "FL" : "REF"}  ${s.state.width}\u00d7${s.state.height}  ${dur}`));

      const ctl = el("div", "ctl");
      const up = el("button", null, "\u25B2"); up.title = "Move up";
      const dn = el("button", null, "\u25BC"); dn.title = "Move down";
      const rm = el("button", "rm", "\u00D7"); rm.title = "Delete this clip";
      up.onclick = (e) => { e.stopPropagation(); moveShot(i, -1); };
      dn.onclick = (e) => { e.stopPropagation(); moveShot(i, 1); };
      rm.onclick = (e) => { e.stopPropagation(); delShot(i); };
      ctl.append(up, dn, rm);

      row.append(th, mid, ctl);
      row.onclick = () => switchTo(i);
      if (i === shotsFocus) row.dataset.focus = "1";
      list.append(row);
    });
    shotsPanel.append(list);
    list.scrollTop = prevTop;
    if (shotsFocus >= 0) {
      const target = list.querySelector('[data-focus="1"]');
      if (target) target.scrollIntoView({ block: "nearest" });
      shotsFocus = -1;
    }

    const foot = el("div", "gcast-shots-foot");
    const bAdd = el("button", "gcast-btn", "Add clip");
    bAdd.title = "New clip keeping the current references, canvas, length and subject_definitions";
    const bBlank = el("button", "gcast-btn ghost", "Blank");
    bBlank.title = "New empty clip";
    const bImport = el("button", "gcast-btn ghost", "Import\u2026");
    bImport.title = "Add a saved .h3.json or .h3pack to this project as a clip";
    const bPRevert = el("button", "gcast-btn ghost revert", "Revert");
    const bPNew = el("button", "gcast-btn ghost", "New");
    const bPSave = el("button", "gcast-btn ghost", "Save project");
    const bPSaveAs = el("button", "gcast-btn ghost", "As\u2026");
    const bPPack = el("button", "gcast-btn ghost", "Pack\u2026");
    const bPOpen = el("button", "gcast-btn ghost", "Open\u2026");
    const bak = prevProject();
    bPRevert.disabled = !bak;
    bPRevert.title = bak
      ? `Go back to the project as it was ${prevWhen(bak)} \u2014 `
        + `${bak.shots.length} clip${bak.shots.length > 1 ? "s" : ""}. `
        + "What you have now is kept, so Revert again brings it back."
      : "Nothing to go back to yet \u2014 a copy is put aside before New, Open, Import "
        + "and Delete, and after every save";
    bPNew.title = "Close this project and start an empty one \u2014 the clip on screen is kept";
    bPSave.title = projLabel
      ? "Write the whole project back over " + projLabel
      : "Write the whole project to one file";
    bPSaveAs.title = "Write the project to a new file";
    bPPack.title = "Export the whole project as one .zip with all the media inside \u2014 shared references are stored once";
    bPOpen.title = "Open a project file";
    bAdd.onclick = (e) => { e.stopPropagation(); addShot(false); };
    bBlank.onclick = (e) => { e.stopPropagation(); addShot(true); };
    bImport.onclick = (e) => { e.stopPropagation(); importShot(); };
    bPRevert.onclick = (e) => { e.stopPropagation(); revertProject(); };
    bPNew.onclick = (e) => { e.stopPropagation(); newProject(); };
    bPSave.onclick = (e) => { e.stopPropagation(); saveProject(false); };
    bPSaveAs.onclick = (e) => { e.stopPropagation(); saveProject(true); };
    bPPack.onclick = (e) => { e.stopPropagation(); packProject(); };
    bPOpen.onclick = (e) => { e.stopPropagation(); openProject(); };
    foot.append(bAdd, bBlank, bImport);
    shotsPanel.append(foot);

    /* Shot actions and project actions on separate rows: seven buttons on one
     * wrapping row read as one undifferentiated pile, and "New" sitting next
     * to "Add clip" is exactly the confusion to avoid. */
    const pfoot = el("div", "gcast-shots-foot proj");
    pfoot.append(el("div", "lbl", "Project"), el("div", "spacer"),
                 bPRevert, bPNew, bPSave, bPSaveAs, bPPack, bPOpen);
    shotsPanel.append(pfoot);
  }

  bShots.onclick = (e) => {
    e.stopPropagation();
    shotsOpen ? closeShots() : openShots();
  };
  paintShotsBtn();

  /* The panel is fixed to the viewport while the button moves with the
   * canvas, so it re-anchors every frame while open. Cheaper than trying to
   * hook LiteGraph's pan and zoom, and it cannot get out of sync. */
  (function followShots() {
    if (shotsOpen) placeShots();
    requestAnimationFrame(followShots);
  })();

  /* Insurance: the frontend claims wheel at window level, which is why the
   * prompt box will not scroll. Same trap would hit a long shot list, so the
   * list is scrolled by hand whenever the pointer is genuinely over it. */
  window.addEventListener("wheel", (e) => {
    if (!shotsOpen || !shotsPanel.contains(e.target)) return;
    const list = shotsPanel.querySelector(".gcast-shots-list");
    if (!list || list.scrollHeight <= list.clientHeight) return;
    list.scrollTop += e.deltaY;
    e.preventDefault(); e.stopPropagation();
  }, { capture: true, passive: false });

  /* The panel's collapsed height, measured rather than guessed. Floating the
   * prompt takes it out of the flow, which would otherwise shrink the panel
   * and open a gap above it in the v2 frontend. */
  let contentH = CONTENT_MIN;
  function measureContent() {
    requestAnimationFrame(() => {
      if (node.properties?.gcast_prompt_big && st.mode !== "fl2va") return;
      const prev = root.style.minHeight;
      root.style.minHeight = "";
      const h = root.scrollHeight;
      root.style.minHeight = prev;
      if (h > contentH) {
        contentH = h;
        root.style.minHeight = contentH + "px";
      }
    });
  }

  /* Anchor the floating prompt just under the last media row, so it covers
   * the audio slots and the strip but leaves the references visible. */
  /* The expanded prompt starts just under the IMAGE row, so it covers the
   * video and audio racks as well as the two strips. Anchoring it under the
   * video row instead left it short: the references you actually check while
   * writing are the images, and the video and audio slots are the ones you
   * set once and stop looking at. */
  function placeLayer() {
    requestAnimationFrame(() => {
      if (!pWrap.classList.contains("gcast-promptlayer")) return;

      const anchor = st.mode === "fl2va" ? flGrid : imgGrid;
      const top = anchor && anchor.offsetHeight
        ? anchor.offsetTop + anchor.offsetHeight + 10
        : 120;
      pWrap.style.top = top + "px";
      syncGutter();
    });
  }

  bExpand.onclick = () => {
    node.properties = node.properties || {};
    node.properties.gcast_prompt_big = !node.properties.gcast_prompt_big;
    render();
    if (node.properties.gcast_prompt_big) setTimeout(() => ta.focus(), 0);
  };

  bSave.onclick = () => saveOver();
  bSaveAs.onclick = () => saveAs(false);
  bPack.onclick = () => saveAs(true);
  bLoad.onclick = () => doLoad();

  function fileForToken(token) {
    if (!token) return null;
    const m = /^@image(\d)$/.exec(token); if (m) return st.slots.images[+m[1] - 1].file;
    const v = /^@video(?:audio)?(\d)$/.exec(token); if (v) return st.slots.videos[+v[1] - 1].file;
    const a = /^@audio(\d)$/.exec(token); if (a) return null;
    if (token === "@first") return st.slots.first.file;
    if (token === "@last") return st.slots.last.file;
    return null;
  }

  const tokenOf = (r) =>
    r.token || (r.from === "first frame" ? "@first" : r.from === "last frame" ? "@last" : null);

  /* ---- @ autocomplete ---------------------------------------------- */

  const ac = el("div", "gcast-ac");
  ac.style.display = "none";
  ac.addEventListener("pointerdown", (e) => e.preventDefault());   // keep textarea focus
  ac.style.position = "fixed";
  document.body.append(ac);   // outside the node, so opening it can't reflow the layout

  let acItems = [], acIdx = 0, acStart = -1;
  let chipSig = "";
  const chipEls = new Map();

  /* caret position, measured with a style-cloned mirror */
  function caretXY() {
    const cs = getComputedStyle(ta);
    const mirror = el("div");
    const copy = ["fontFamily", "fontSize", "fontWeight", "lineHeight", "letterSpacing",
      "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
      "borderTopWidth", "borderLeftWidth", "textTransform", "wordSpacing"];
    copy.forEach((k) => { mirror.style[k] = cs[k]; });
    Object.assign(mirror.style, {
      position: "absolute", visibility: "hidden", whiteSpace: "pre-wrap",
      wordWrap: "break-word", top: "0", left: "0",
      width: ta.clientWidth + "px", boxSizing: "border-box",
    });
    Object.assign(mirror.style, { left: "-99999px", top: "0" });
    const head = document.createTextNode(ta.value.slice(0, ta.selectionStart));
    const mark = el("span", null, "\u200b");
    mirror.append(head, mark);
    document.body.append(mirror);
    const x = mark.offsetLeft, y = mark.offsetTop;
    mirror.remove();
    const box = ta.getBoundingClientRect();
    return { x: box.left + x, y: box.top + y - ta.scrollTop + parseFloat(cs.lineHeight || 18) + 4 };
  }

  function closeAC() { ac.style.display = "none"; acItems = []; acStart = -1; acSignature = ""; }

  /* The @ list also offers the next shot marker.
   *
   * The bracket-and-timestamp format is the one thing in an H3 prompt you
   * cannot guess, and typing "[Shot 2] At 00:04.000," by hand is exactly where
   * people give up and write prose instead. So the menu writes it.
   *
   * Shot 1 carries no timestamp - that is the format's rule, not a shortcut.
   * Later markers default to halfway between the previous one and the end of
   * the clip, rounded to the half second: a placeholder you can immediately
   * see on the timeline strip, instead of a 00:00 that stacks everything on
   * top of itself. */
  function mmss(t) {
    const m = Math.floor(t / 60), sec = Math.floor(t % 60);
    const ms = Math.round((t - Math.floor(t)) * 1000);
    return `${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}.${String(ms).padStart(3, "0")}`;
  }

  function nextShotMarker() {
    const total = Math.max(0.001, (alignFrames(st.length) - 1) / FPS);
    const found = parseShots(st.prompt || "", total);
    const n = found.length + 1;
    if (n === 1) return { text: "[Shot 1] ", token: "[Shot 1]", tag: "starts the clip" };
    const timed = found.filter((x) => !x.untimed).map((x) => x.start);
    const last = timed.length ? Math.max.apply(null, timed) : 0;
    let t = last + (total - last) / 2;
    t = Math.max(0, Math.min(total, Math.round(t * 2) / 2));
    return { text: `[Shot ${n}] At ${mmss(t)}, `, token: `[Shot ${n}]`, tag: `At ${mmss(t)}` };
  }

  function openAC(query, start) {
    const { rows } = presentation(st);
    const q = query.toLowerCase();
    acItems = rows
      .map((r) => ({ token: tokenOf(r), tag: r.tag, from: r.from, kind: r.kind }))
      .filter((o) => o.token && (!q || o.token.slice(1).toLowerCase().includes(q)));

    /* always last in the list - it is an action, not a reference */
    if (!q || "shot marker cut".includes(q)) {
      const mk = nextShotMarker();
      acItems.push({ special: "shot", insert: mk.text, token: mk.token, tag: mk.tag, kind: "shot" });
    }
    acStart = start;
    acIdx = 0;
    if (!acItems.length) { closeAC(); return; }
    ac.dataset.mode = st.mode;
    drawAC();
    const { x, y } = caretXY();
    ac.style.display = "";
    const w = ac.offsetWidth || 220, h = ac.offsetHeight || 200;
    ac.style.left = Math.max(6, Math.min(x, window.innerWidth - w - 8)) + "px";
    ac.style.top = (y + h > window.innerHeight - 8 ? Math.max(6, y - h - 26) : y) + "px";
  }

  let acSignature = "";

  function markAC() {
    Array.from(ac.children).forEach((b, i) => {
      if (b.tagName === "BUTTON") b.setAttribute("aria-selected", String(i === acIdx));
    });
  }

  function drawAC() {
    const sig = acItems.map((o) => o.token + o.tag).join("|");
    if (sig === acSignature && ac.children.length) { markAC(); return; }
    acSignature = sig;

    ac.replaceChildren();
    if (!acItems.length) {
      ac.append(el("div", "none", "No references filled yet"));
      return;
    }
    acItems.forEach((o, i) => {
      const b = el("button");
      const file = fileForToken(o.token);
      if (o.special) b.append(el("span", "glyph mark", "\u2702"));
      else if (o.kind === "image" && file) { const im = el("img"); im.src = viewURL(file); b.append(im); }
      else if (o.kind === "video" && file) { const v = el("video"); v.src = viewURL(file); v.muted = true; v.preload = "metadata"; b.append(v); }
      else b.append(el("span", "glyph", "\u266a"));
      b.append(el("span", null, o.token), el("span", "tag", o.tag));
      b.addEventListener("mouseenter", () => { acIdx = i; markAC(); });
      b.addEventListener("pointerdown", (e) => { e.preventDefault(); e.stopPropagation(); acceptAC(i); });
      ac.append(b);
    });
    markAC();
  }

  function acceptAC(i) {
    const o = acItems[i];
    if (!o) { closeAC(); return; }
    const caret = ta.selectionStart;
    const before = ta.value.slice(0, acStart);

    /* a shot marker starts a line; a reference tag goes wherever you are */
    const insert = o.special
      ? ((before.length && !/\n[ \t]*$/.test(before) ? "\n" : "") + o.insert)
      : o.token + " ";

    ta.value = before + insert + ta.value.slice(caret);
    const pos = acStart + insert.length;
    ta.setSelectionRange(pos, pos);
    ta.focus();
    st.prompt = ta.value;
    closeAC(); renderTags(); commit();
    if (o.special) renderCheck();
  }

  function maybeAC() {
    const caret = ta.selectionStart;
    const m = /@([A-Za-z0-9]*)$/.exec(ta.value.slice(0, caret));
    if (!m) { closeAC(); return; }
    openAC(m[1], caret - m[0].length);
  }

  ta.addEventListener("keydown", (e) => {
    if (ac.style.display === "none") return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault(); e.stopPropagation();
      acIdx = (acIdx + (e.key === "ArrowDown" ? 1 : -1) + acItems.length) % Math.max(1, acItems.length);
      markAC();
      ac.children[acIdx]?.scrollIntoView?.({ block: "nearest" });
    } else if (e.key === "Enter" || e.key === "Tab") {
      if (!acItems.length) return;
      e.preventDefault(); e.stopPropagation();
      acceptAC(acIdx);
    } else if (e.key === "Escape") {
      e.preventDefault(); e.stopPropagation();
      closeAC();
    }
  });
  ta.addEventListener("blur", () => setTimeout(closeAC, 120));
  ta.addEventListener("click", () => maybeAC());

  function insert(token) {
    const s = ta.selectionStart ?? ta.value.length;
    const e = ta.selectionEnd ?? s;
    const pad = (s > 0 && !/\s$/.test(ta.value.slice(0, s))) ? " " : "";
    ta.value = ta.value.slice(0, s) + pad + token + " " + ta.value.slice(e);
    const pos = s + pad.length + token.length + 1;
    ta.focus(); ta.setSelectionRange(pos, pos);
    st.prompt = ta.value; commit();
  }

  /* -------------------------------------------------------- listeners */

  /* pointerdown rather than click: something in the v2 node renderer can
   * swallow the click before it lands, and stopping propagation here keeps
   * the canvas-forwarding handler on root out of the way. */
  const setMode = (m) => {
    if (st.mode === m) return;
    st.mode = m;
    render();
    commit();
  };
  const wireMode = (btn, mode) => {
    btn.addEventListener("pointerdown", (e) => {
      e.preventDefault(); e.stopPropagation();
      setMode(mode);
    });
    btn.addEventListener("click", (e) => { e.preventDefault(); e.stopPropagation(); });
  };
  wireMode(btnFL, "fl2va");
  wireMode(btnRef, "ref2va");

  selRatio.onchange = (v) => {
    if (v === "custom") return;
    const fam = RATIOS.find((r) => r.label === v);
    if (!fam) return;
    /* keep the rung of the ladder they were already on, if it exists */
    const cur = findRatio(st.width, st.height);
    const rung = cur ? cur.sizes.findIndex((z) => z.w === st.width && z.h === st.height) : 0;
    const pick = fam.sizes[Math.min(Math.max(0, rung), fam.sizes.length - 1)];
    st.width = pick.w; st.height = pick.h; render(); commit();
  };
  selSize.onchange = (v) => {
    if (v === "custom") return;
    const [w, h] = v.split("x").map(Number);
    st.width = w; st.height = h; render(); commit();
  };
  const snap = (v) => Math.max(32, Math.round(v / 32) * 32);
  inW.onchange = () => { st.width = snap(+inW.value || 768); render(); commit(); };
  inH.onchange = () => { st.height = snap(+inH.value || 768); render(); commit(); };

  selLen.onchange = (v) => {
    if (v === "custom") return;
    st.length = +v; render(); commit();
  };
  inLen.onchange = () => { st.length = alignFrames(+inLen.value || 124); render(); commit(); };

  bMatch.onclick = () => { st.ref_image_size = "match"; render(); commit(); };
  bMax.onclick = () => { st.ref_image_size = "max"; render(); commit(); };

  ta.addEventListener("input", () => { st.prompt = ta.value; renderTags(); renderCheck(); maybeAC(); commit(); });
  ta.addEventListener("pointerdown", (e) => e.stopPropagation());
  /* The DOM widget sits over the canvas and eats wheel and drag. Rather than
   * disabling pointer events (which risks reaching a shared container and
   * freezing the whole graph), re-dispatch the event onto the canvas. */
  const graphCanvas = () => app.canvas?.canvas || document.querySelector("canvas#graph-canvas");

  const INTERACTIVE = "button, select, input, textarea, label, " +
    ".gcast-slot, .gcast-media, .gcast-chip, .gcast-track, .gcast-thumb, .gcast-wav";

  root.addEventListener("wheel", (e) => {
    closeAC();
    /* Anything that can scroll itself keeps the wheel -- but only while it
     * actually has somewhere to go. Returning early is not enough on its own:
     * without stopPropagation the event carries on to the canvas handler and
     * zooms the graph instead of scrolling the prompt. At either end of the
     * scroll, hand it on so a short prompt still zooms like the rest of the
     * panel. No preventDefault -- the browser does the scrolling. */
    /* A scrollable area keeps the wheel outright. Handing it on at the ends
     * sounds tidy but means a wheel-down at scrollTop 0 zooms the graph
     * instead of scrolling -- which is the whole complaint. overscroll-behavior
     * in the CSS stops the scroll chaining onward once it does hit an end. */
    if (e.target.closest("textarea, .gcast-ac")) {
      e.stopPropagation();
      return;
    }
    const cv = graphCanvas();
    if (!cv) return;
    e.preventDefault();
    e.stopPropagation();
    cv.dispatchEvent(new WheelEvent("wheel", {
      deltaX: e.deltaX, deltaY: e.deltaY, deltaMode: e.deltaMode,
      clientX: e.clientX, clientY: e.clientY,
      ctrlKey: e.ctrlKey, shiftKey: e.shiftKey,
      bubbles: true, cancelable: true,
    }));
  }, { passive: false });

  /* NO pointerdown forwarding to the canvas. v2 takes pointer CAPTURE on that
   * press and only releases it from its own state machine, which ignores an
   * untrusted event -- so the canvas keeps panning with the mouse until a
   * real press elsewhere clears it. Confirmed in the event log: every stuck
   * drag began with a re-dispatched pointerdown and ended on the next real
   * press with lostpointercapture. Wheel forwarding above is fine, because
   * wheel does not capture. The node still drags by its title bar.
   */

  /* Miss a slot and the file falls through to ComfyUI, which answers by
   * spawning a LoadAudio node on the graph. The node swallows the whole drop
   * instead: while a file is over it, every slot that could take that file is
   * armed so the target is visible, and a release that lands on none of them
   * is routed to the first empty slot of the right kind rather than escaping. */
  let dragDepth = 0;

  function armSlots(kind) {
    root.querySelectorAll("[data-kind]").forEach((n) => {
      n.classList.toggle("gcast-armed", kind === "" || n.dataset.kind === kind);
    });
  }
  function disarm() {
    dragDepth = 0;
    root.querySelectorAll(".gcast-armed, .drop").forEach((n) => {
      n.classList.remove("gcast-armed", "drop");
    });
  }

  root.addEventListener("dragenter", (e) => {
    const k = dragKind(e);
    if (k === null) return;
    e.preventDefault(); e.stopPropagation();
    dragDepth++;
    armSlots(k);
  });
  root.addEventListener("dragover", (e) => {
    if (dragKind(e) === null) return;
    e.preventDefault(); e.stopPropagation();      /* keep it off the canvas */
  });
  root.addEventListener("dragleave", (e) => {
    if (dragKind(e) === null) return;
    if (--dragDepth <= 0) disarm();
  });
  window.addEventListener("dragend", disarm);
  window.addEventListener("drop", disarm);

  root.addEventListener("drop", async (e) => {
    const k = dragKind(e);
    if (k === null) return;
    e.preventDefault(); e.stopPropagation();       /* never reaches the graph */
    const onSlot = e.target.closest("[data-kind]");
    disarm();
    if (onSlot) return;                            /* the slot handled it */
    const f = e.dataTransfer?.files?.[0];
    const kind = f && (fileKind(f) || k);
    if (!f || !kind) return;
    const routed = routeToFreeSlot(kind, f);
    if (!routed) flashPanel();
  });

  ac.addEventListener("wheel", (e) => e.stopPropagation());
  root.addEventListener("scroll", () => closeAC(), true);

  function load(raw) { st = parseInitial(raw); ta.value = st.prompt; render(); }

  return {
    root,
    destroy() { ac.remove(); },
    load,
    save() { return JSON.stringify(st); },
    get state() { return st; },
  };
}

/* ============================================================ register */

app.registerExtension({
  name: "cglide.glidecast",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_ID) return;
    injectCSS();

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onCreated?.apply(this, arguments);

      const data = this.widgets?.find((w) => w.name === "h3_data");
      if (data) {
        /* Hiding a widget differs between frontends:
         *   v1 honours computeSize returning zero height
         *   v2 honours widget.hidden, and does NOT understand type "hidden" --
         *     forcing that type makes it fall back to drawing a plain text
         *     widget, which then floats over the panel and eats clicks.
         * So: set both real mechanisms, and never touch .type. */
        /* Hiding a widget has changed convention across frontend versions, so
         * set every mechanism that has ever been honoured:
         *   v1  - computeSize returning zero height
         *   v2  - widget.hidden / computeLayoutSize
         *   both - type "converted-widget", the convention used when a widget
         *          is turned into an input socket. It is a type the frontend
         *          KNOWS, unlike "hidden", which it drew as a plain text row. */
        data.origType = data.origType || data.type;
        data.origComputeSize = data.computeSize;
        data.type = "converted-widget";
        data.hidden = true;
        data.options = Object.assign({}, data.options, { hidden: true });
        data.computeSize = () => [0, -4];
        data.computeLayoutSize = () => ({ minHeight: 0, maxHeight: 0, minWidth: 0 });
        data.label = "";
        data.tooltip = "";
        data.serializeValue = data.serializeValue || (() => data.value);
        if (data.element) {
          Object.assign(data.element.style, {
            display: "none", visibility: "hidden", pointerEvents: "none",
            position: "absolute", left: "-99999px", top: "0",
            width: "0px", height: "0px", opacity: "0", zIndex: "-1",
          });
          data.element.hidden = true;
          data.element.tabIndex = -1;
        }
      }

      const ui = buildUI(this);
      this.h3ui = ui;
      const uiWidget = this.addDOMWidget("h3_ui", "div", ui.root, { serialize: false, hideOnZoom: false });

      /* The widget's own wrapper is <div class="dom-widget">, sitting inside
       * .isolate inside #graph-canvas-container. ONLY the dom-widget div may be
       * made transparent -- going any higher reaches the canvas container and
       * freezes the entire graph. */
      requestAnimationFrame(() => {
        const wrap = ui.root.parentElement;
        if (wrap && wrap.classList.contains("dom-widget")) {
          wrap.style.pointerEvents = "none";
        }
      });

      /* v2 wraps every DOM widget in a full-size .dom-widget div. Hiding the
       * inner element is not enough -- the wrapper stays, covers the node and
       * swallows every click. Hide the wrappers belonging to THIS node's other
       * widgets only; never touch shared containers. */
      const hideStrayWrappers = () => {
        const mine = ui.root.closest(".dom-widget");
        for (const w of this.widgets || []) {
          if (!w || w === uiWidget || !w.element) continue;
          const wrap = w.element.closest?.(".dom-widget");
          if (wrap && wrap !== mine) {
            wrap.style.display = "none";
            wrap.style.pointerEvents = "none";
          }
        }
      };
      requestAnimationFrame(hideStrayWrappers);
      setTimeout(hideStrayWrappers, 250);
      this.gcastTidy = hideStrayWrappers;

      ui.load(data ? data.value : "");

      this.size = [980, MIN_H_REF];
      this.setSize?.([980, MIN_H_REF]);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.h3ui?.destroy?.();
      return onRemoved?.apply(this, arguments);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      const r = onConfigure?.apply(this, arguments);
      const data = this.widgets?.find((w) => w.name === "h3_data");
      if (this.h3ui && data) this.h3ui.load(data.value);
      if (this.gcastTidy) setTimeout(this.gcastTidy, 60);
      return r;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const minH = this.h3MinHeight || MIN_H_REF;
      if (size[0] < 900) size[0] = 900;
      if (size[1] < minH) size[1] = minH;
      return onResize?.apply(this, arguments);
    };
  },
});
