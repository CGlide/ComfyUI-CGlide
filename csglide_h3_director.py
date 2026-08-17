"""Glide H3 Director — one node covering both MiniMax H3 model families.

  mode "fl2va"  -> first / last keyframes      (needs minimax_h3_fl2va_* weights)
  mode "ref2va" -> 9 images / 3 videos / 3 audio (needs minimax_h3_ref2va_* weights)

Everything the UI holds lives in the single `h3_data` JSON widget. Reference tags
are written in the prompt as @image1 / @video1 / @audio1 and transcribed here to
the real <Picture i> / <Video k> / <Audio j> presentation, renumbered against the
slots that are actually filled.
"""

import json
import math
import os
import shutil
import time
import zipfile

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
import nodes
import comfy.model_management
import comfy.nested_tensor
import comfy.utils
import node_helpers

try:
    import av
except Exception:  # pragma: no cover
    av = None

try:
    import torchaudio
except Exception:  # pragma: no cover
    torchaudio = None


# --------------------------------------------------------------------------
# H3 constants (mirrors comfy_extras/nodes_minimax_h3.py)
# --------------------------------------------------------------------------

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40

ASSET_SUBFOLDER = "cglide"
LEGACY_ASSET_SUBFOLDERS = ("whatdreamscost",)

MAX_IMAGES = 9
MAX_VIDEOS = 3
MAX_AUDIOS = 3


def align_frame_count(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(length)
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def adapt_canvas(width, height):
    """768-short-edge canvas with a 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


# --------------------------------------------------------------------------
# Asset resolution + loading
# --------------------------------------------------------------------------

def _resolve_asset(ref):
    """Find an uploaded asset. Tries the ref as given, then each asset folder."""
    if not ref:
        return None
    root = folder_paths.get_input_directory()
    candidates = [os.path.join(root, ref)]
    base = os.path.basename(ref)
    for sub in (ASSET_SUBFOLDER,) + LEGACY_ASSET_SUBFOLDERS:
        candidates.append(os.path.join(root, sub, base))
    candidates.append(os.path.join(root, base))
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    return None


def _load_image(ref):
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Director: image not found on disk: %s" % ref)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _load_video_frames(ref, start, end, max_frames):
    """Decode [start, end) seconds and resample to FPS. Returns [N, H, W, 3]."""
    if av is None:
        raise RuntimeError("H3 Director: PyAV is required to read reference videos.")
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Director: video not found on disk: %s" % ref)

    frames, times = [], []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base else 1.0 / FPS
        if start > 0:
            try:
                container.seek(int(max(0.0, start - 0.5) / tb), stream=stream)
            except Exception:
                pass
        for frame in container.decode(video=0):
            t = float(frame.pts * tb) if frame.pts is not None else (len(frames) / FPS)
            if t < start - 1e-4:
                continue
            if end is not None and t > end + 1e-4:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(t)
            if len(frames) > 4096:
                break

    if not frames:
        raise ValueError("H3 Director: no frames decoded from %s in that trim range." % ref)

    span = (times[-1] - times[0]) if len(times) > 1 else 0.0
    want = max(1, int(round(span * FPS)) + 1)
    want = min(want, max_frames)
    idx, cur = [], 0
    for k in range(want):
        target = times[0] + k / FPS
        while cur + 1 < len(times) and abs(times[cur + 1] - target) <= abs(times[cur] - target):
            cur += 1
        idx.append(cur)

    arr = np.stack([frames[i] for i in idx]).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_audio(ref, start, end):
    """Returns a ComfyUI AUDIO dict, trimmed to [start, end)."""
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Director: audio not found on disk: %s" % ref)

    waveform, sr = None, None
    if torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            waveform = None
    if waveform is None:
        if av is None:
            raise RuntimeError("H3 Director: cannot read audio (no torchaudio, no PyAV).")
        chunks = []
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError("H3 Director: %s has no audio stream." % ref)
            stream = container.streams.audio[0]
            sr = stream.rate
            for frame in container.decode(audio=0):
                chunks.append(frame.to_ndarray())
        data = np.concatenate(chunks, axis=-1) if chunks else np.zeros((1, 0), np.float32)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim == 1:
            data = data[None, :]
        waveform = torch.from_numpy(data)

    if waveform.ndim == 1:
        waveform = waveform[None, :]
    a = int(max(0.0, start) * sr)
    b = int(end * sr) if end is not None else waveform.shape[-1]
    b = min(b, waveform.shape[-1])
    if b <= a:
        raise ValueError("H3 Director: audio trim for %s is empty." % ref)
    waveform = waveform[..., a:b]
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    return {"waveform": waveform[None, ...], "sample_rate": int(sr)}


def _load_video_soundtrack(ref, start, end):
    try:
        return _load_audio(ref, start, end)
    except Exception as e:
        print("[H3 Director] no usable soundtrack in %s (%s) — video sent silent." % (ref, e))
        return None


# --------------------------------------------------------------------------
# Slot parsing + tag transcription
# --------------------------------------------------------------------------

def _slot_list(raw, count):
    out = []
    src = raw if isinstance(raw, list) else []
    for i in range(count):
        item = src[i] if i < len(src) and isinstance(src[i], dict) else {}
        out.append(item if item.get("file") else None)
    return out


def parse_h3_data(raw):
    """Whitelisted parse — anything not listed here is dropped on purpose."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    slots = data.get("slots") if isinstance(data.get("slots"), dict) else {}

    def one(key):
        item = slots.get(key)
        return item if isinstance(item, dict) and item.get("file") else None

    return {
        "mode": "fl2va" if data.get("mode") == "fl2va" else "ref2va",
        "width": int(data.get("width") or 1344),
        "height": int(data.get("height") or 768),
        "length": int(data.get("length") or 124),
        "ref_image_size": "max" if data.get("ref_image_size") == "max" else "match",
        "prompt": str(data.get("prompt") or ""),
        "first": one("first"),
        "last": one("last"),
        "images": _slot_list(slots.get("images"), MAX_IMAGES),
        "videos": _slot_list(slots.get("videos"), MAX_VIDEOS),
        "audios": _slot_list(slots.get("audios"), MAX_AUDIOS),
    }


def build_tag_map(cfg, has_first, has_last):
    """Map @token -> real tag, using the same ordinal rules as the tokenizer.

    Presentation order is fixed by type: images, then videos (a soundtrack's
    <Audio j> is emitted immediately before its <Video k>), then standalone
    audio. Ordinals are 1-based per type and count only filled slots.
    """
    tags, presentation = {}, []

    if cfg["mode"] == "fl2va":
        i = 0
        if has_first:
            i += 1
            tags["@first"] = "<Picture %d>" % i
            presentation.append(("<Picture %d>" % i, "first frame"))
        if has_last:
            i += 1
            tags["@last"] = "<Picture %d>" % i
            presentation.append(("<Picture %d>" % i, "last frame"))
        return tags, presentation

    i = 0
    for n, slot in enumerate(cfg["images"], start=1):
        if slot is None:
            continue
        i += 1
        tags["@image%d" % n] = "<Picture %d>" % i
        presentation.append(("<Picture %d>" % i, "image %d" % n))

    j, k = 0, 0
    for n, slot in enumerate(cfg["videos"], start=1):
        if slot is None:
            continue
        if slot.get("audio"):
            j += 1
            tags["@videoaudio%d" % n] = "<Audio %d>" % j
            presentation.append(("<Audio %d>" % j, "video %d sound" % n))
        k += 1
        tags["@video%d" % n] = "<Video %d>" % k
        presentation.append(("<Video %d>" % k, "video %d" % n))

    for n, slot in enumerate(cfg["audios"], start=1):
        if slot is None:
            continue
        j += 1
        tags["@audio%d" % n] = "<Audio %d>" % j
        presentation.append(("<Audio %d>" % j, "audio %d" % n))

    return tags, presentation


def transcribe_prompt(prompt, tags, known_tokens):
    """Replace @tokens with real tags. Tokens pointing at empty slots are dropped."""
    out = prompt
    for token in sorted(known_tokens, key=len, reverse=True):
        if token in tags:
            out = out.replace(token, tags[token])
        elif token in out:
            print("[H3 Director] %s points at an empty slot — removed from the prompt." % token)
            out = out.replace(token, "")
    return out


# --------------------------------------------------------------------------
# Presets: save / save as / save packed / load
# --------------------------------------------------------------------------
#
# Presets live beside ComfyUI's user data rather than inside the workflow, so
# one can be loaded into a fresh node in any graph.
#
# Two shapes:
#   .json    state only, media referenced by name. Small, instant, and dead
#            if input/cglide/ is ever cleared.
#   .h3pack  zip holding the same state plus the media itself. Survives a
#            move to another machine; loading restores anything missing.

PRESET_EXT = ".json"
PACK_EXT = ".h3pack"


def _preset_dir():
    if folder_paths is not None and hasattr(folder_paths, "get_user_directory"):
        root = folder_paths.get_user_directory()
    else:
        root = os.path.join(os.path.abspath("user"))
    path = os.path.join(root, "cglide_h3")
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(name):
    """One path component, no traversal, no separators."""
    name = os.path.basename(str(name or "").strip())
    keep = "-_. ()[]"
    name = "".join(c for c in name if c.isalnum() or c in keep).strip()
    # "." and ".." survive the filter above, so reject dot-only components
    # outright rather than leaning on the realpath guard alone
    if not name.strip("."):
        return "untitled"
    return name[:120] or "untitled"


def _safe_rel(rel):
    """A project-folder path under the preset dir. Components sanitised
    individually, depth capped, ".." impossible by construction."""
    parts = [p for p in str(rel or "").replace("\\", "/").split("/") if p.strip()]
    parts = [_safe_name(p) for p in parts][-4:]     # at most 3 folders + name
    return parts or ["untitled"]


def _inside(path):
    """Guard: the resolved path must still be under the preset dir."""
    root = os.path.realpath(_preset_dir())
    full = os.path.realpath(path)
    return full == root or full.startswith(root + os.sep)


def list_folders():
    """Existing project folders, relative and slash-separated."""
    root = _preset_dir()
    out = set()
    for dirpath, dirnames, _ in os.walk(root):
        for d in dirnames:
            rel = os.path.relpath(os.path.join(dirpath, d), root).replace(os.sep, "/")
            out.add(rel)
    return sorted(out)


def _asset_refs(data):
    """Every media reference in a saved state, as (container, key) pairs."""
    slots = data.get("slots") or {}
    out = []
    for key in ("first", "last"):
        item = slots.get(key)
        if isinstance(item, dict) and item.get("file"):
            out.append(item)
    for key in ("images", "videos", "audios"):
        for item in (slots.get(key) or []):
            if isinstance(item, dict) and item.get("file"):
                out.append(item)
    return out


def _preset_path(name, packed):
    parts = _safe_rel(name)
    parts[-1] = parts[-1] + (PACK_EXT if packed else PRESET_EXT)
    path = os.path.join(_preset_dir(), *parts)
    if not _inside(path):
        raise ValueError("bad preset path")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_preset(name, data, packed=False, folder=""):
    """Write a preset, optionally inside a project folder.

    `name` may itself contain slashes; `folder` is prepended if given.
    """
    if isinstance(data, str):
        data = json.loads(data or "{}")
    rel = ("%s/%s" % (folder, name)) if folder else name
    parts = _safe_rel(rel)
    path = _preset_path(rel, packed)

    meta = {
        "name": parts[-1],
        "folder": "/".join(parts[:-1]),
        "mode": "fl2va" if data.get("mode") == "fl2va" else "ref2va",
        "packed": bool(packed),
        "saved": time.time(),
        "width": data.get("width"),
        "height": data.get("height"),
        "length": data.get("length"),
    }

    if not packed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "state": data}, f, indent=1)
        return os.path.relpath(path, _preset_dir()).replace(os.sep, "/")

    missing = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("meta.json", json.dumps(meta, indent=1))
        z.writestr("state.json", json.dumps(data, indent=1))
        seen = set()
        for item in _asset_refs(data):
            ref = item["file"]
            src = _resolve_asset(ref)
            base = os.path.basename(ref)
            if src is None:
                missing.append(ref)
                continue
            if base in seen:
                continue
            seen.add(base)
            z.write(src, "assets/" + base)
    if missing:
        print("[H3 Director] packed %s, but these were not on disk: %s"
              % (os.path.basename(path), ", ".join(missing)))
    return os.path.relpath(path, _preset_dir()).replace(os.sep, "/")


def list_presets():
    """Every preset in every project folder, newest first."""
    root = _preset_dir()
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            try:
                if fn.endswith(PACK_EXT):
                    with zipfile.ZipFile(path) as z:
                        meta = json.loads(z.read("meta.json").decode("utf-8"))
                elif fn.endswith(PRESET_EXT):
                    with open(path, encoding="utf-8") as f:
                        meta = (json.load(f) or {}).get("meta") or {}
                else:
                    continue
            except Exception as e:
                print("[H3 Director] skipping unreadable preset %s (%s)" % (fn, e))
                continue
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            meta["file"] = rel
            meta["folder"] = os.path.dirname(rel)
            meta.setdefault("name", os.path.splitext(fn)[0])
            meta.setdefault("mode", "ref2va")
            meta["packed"] = fn.endswith(PACK_EXT)
            meta["bytes"] = os.path.getsize(path)
            out.append(meta)
    out.sort(key=lambda m: m.get("saved") or 0, reverse=True)
    return out


def load_preset(filename):
    """Read a preset. For packs, restore any media that is not on disk."""
    fn = str(filename or "")
    path = os.path.join(_preset_dir(), *_safe_rel(fn))
    if not _inside(path):
        raise ValueError("bad preset path")
    if not os.path.isfile(path):
        raise FileNotFoundError("no such preset: %s" % fn)

    if fn.endswith(PACK_EXT):
        restored = []
        with zipfile.ZipFile(path) as z:
            data = json.loads(z.read("state.json").decode("utf-8"))
            target_dir = os.path.join(folder_paths.get_input_directory(), ASSET_SUBFOLDER) \
                if folder_paths is not None else os.path.abspath("input")
            os.makedirs(target_dir, exist_ok=True)
            for item in _asset_refs(data):
                ref = item["file"]
                base = os.path.basename(ref)
                if _resolve_asset(ref) is not None:
                    continue
                member = "assets/" + base
                if member not in z.namelist():
                    print("[H3 Director] %s references %s but the pack has no copy" % (fn, base))
                    continue
                dest = os.path.join(target_dir, base)
                with z.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                item["file"] = "%s/%s" % (ASSET_SUBFOLDER, base)
                restored.append(base)
        if restored:
            print("[H3 Director] restored %d file(s) from %s: %s"
                  % (len(restored), fn, ", ".join(restored)))
        return data

    with open(path, encoding="utf-8") as f:
        return (json.load(f) or {}).get("state") or {}


def delete_preset(filename):
    path = os.path.join(_preset_dir(), *_safe_rel(str(filename or "")))
    if not _inside(path):
        raise ValueError("bad preset path")
    if os.path.isfile(path):
        os.remove(path)
        # tidy up a project folder that has just become empty
        parent = os.path.dirname(path)
        while _inside(parent) and os.path.realpath(parent) != os.path.realpath(_preset_dir()):
            if os.listdir(parent):
                break
            os.rmdir(parent)
            parent = os.path.dirname(parent)
        return True
    return False


# --- HTTP routes ----------------------------------------------------------
# Namespaced under /cglide/h3/ so nothing can collide with the LTX Director.

try:
    from aiohttp import web
    from server import PromptServer

    _routes = PromptServer.instance.routes

    @_routes.get("/cglide/h3/presets")
    async def _h3_list(request):
        try:
            return web.json_response({"presets": list_presets(),
                                      "folders": list_folders()})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @_routes.post("/cglide/h3/preset")
    async def _h3_save(request):
        try:
            body = await request.json()
            fn = save_preset(body.get("name"), body.get("state") or {},
                             packed=bool(body.get("packed")),
                             folder=body.get("folder") or "")
            return web.json_response({"file": fn})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @_routes.get("/cglide/h3/preset")
    async def _h3_load(request):
        try:
            return web.json_response({"state": load_preset(request.query.get("file"))})
        except FileNotFoundError as e:
            return web.json_response({"error": str(e)}, status=404)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    @_routes.delete("/cglide/h3/preset")
    async def _h3_delete(request):
        try:
            return web.json_response({"deleted": delete_preset(request.query.get("file"))})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

except Exception as _e:  # importing outside ComfyUI, or an older server
    print("[H3 Director] preset routes not registered (%s)" % _e)


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class CSGlideH3Director:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "h3_data": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = ("positive", "latent", "width", "height", "length", "seconds")
    FUNCTION = "build"
    CATEGORY = "CGlide"
    DESCRIPTION = "MiniMax H3 director — first/last keyframes or omni references, with automatic reference tagging."

    # ---------------- reference encoding ----------------

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        if audio_vae is None:
            raise ValueError("H3 Director: audio references need the audio VAE connected "
                             "(minimax_h3_audio_vae_fp32).")
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            if torchaudio is None:
                raise RuntimeError("H3 Director: torchaudio is required to resample reference audio.")
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z, z.shape[-1]

    def _refs(self, cfg, vae, audio_vae, width, height, frame_count):
        ref_items, ref_blocks = [], []
        size_mode = cfg["ref_image_size"]

        for slot in cfg["images"]:
            if slot is None:
                continue
            img = _load_image(slot["file"])
            h, w = img.shape[1], img.shape[2]
            if size_mode == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
                               "latent": vae.encode(resized)})

        for slot in cfg["videos"]:
            if slot is None:
                continue
            start = float(slot.get("start") or 0.0)
            end = slot.get("end")
            end = float(end) if end not in (None, "") else None
            frames = _load_video_frames(slot["file"], start, end, frame_count)

            vh, vw = frames.shape[1], frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("H3 Director: reference videos need at least 5 frames "
                                 "(~0.2s) after trimming — check the trim on %s." % slot["file"])
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)

            audio_latent, ref_audio_t = None, 0
            if slot.get("audio"):
                track = _load_video_soundtrack(slot["file"], start, end)
                if track is not None:
                    audio_latent, ref_audio_t = self._encode_ref_audio(audio_vae, track)
                    ref_items.append({"type": "audio"})

            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            ref_items.append({"type": "video", "data": frames[sample_idx],
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        for slot in cfg["audios"]:
            if slot is None:
                continue
            end = slot.get("end")
            track = _load_audio(slot["file"], float(slot.get("start") or 0.0),
                                float(end) if end not in (None, "") else None)
            audio_latent, ref_audio_t = self._encode_ref_audio(audio_vae, track)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})

        return ref_items, ref_blocks

    # ---------------- main ----------------

    def build(self, clip, vae, h3_data, audio_vae=None, first_frame=None, last_frame=None):
        cfg = parse_h3_data(h3_data)

        width = max(CANVAS_MULTIPLE, (cfg["width"] // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        height = max(CANVAS_MULTIPLE, (cfg["height"] // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        latent, frame_count = _empty_av_latent(width, height, cfg["length"])
        seconds = frame_count / FPS

        if cfg["mode"] == "fl2va":
            first = first_frame if first_frame is not None else (
                _load_image(cfg["first"]["file"]) if cfg["first"] else None)
            last = last_frame if last_frame is not None else (
                _load_image(cfg["last"]["file"]) if cfg["last"] else None)

            tags, _ = build_tag_map(cfg, first is not None, last is not None)
            prompt = transcribe_prompt(cfg["prompt"], tags, ["@first", "@last"])

            images, keyframes = [], []
            if first is not None:
                img = _resize(first[:1], width, height, "disabled")   # geometry anchor: stretch
                images.append(img)
                keyframes.append({"resolved_frame_index": 0, "image": img})
            if last is not None:
                img = _resize(last[:1], width, height, "center")      # follower: cover-crop
                images.append(img)
                keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

            tokens = clip.tokenize(prompt, images=images)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if keyframes:
                for kf in keyframes:
                    kf["latent"] = vae.encode(kf.pop("image"))
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": keyframes,
                    "minimax_frame_count": frame_count,
                })
            return (cond, latent, width, height, frame_count, seconds)

        # ---- ref2va ----
        known = ([f"@image{i}" for i in range(1, MAX_IMAGES + 1)]
                 + [f"@video{i}" for i in range(1, MAX_VIDEOS + 1)]
                 + [f"@videoaudio{i}" for i in range(1, MAX_VIDEOS + 1)]
                 + [f"@audio{i}" for i in range(1, MAX_AUDIOS + 1)])
        tags, presentation = build_tag_map(cfg, False, False)
        prompt = transcribe_prompt(cfg["prompt"], tags, known)

        ref_items, ref_blocks = self._refs(cfg, vae, audio_vae, width, height, frame_count)
        if presentation:
            print("[H3 Director] presentation: " + "  ".join("%s %s" % p for p in presentation))

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return (cond, latent, width, height, frame_count, seconds)

    @classmethod
    def IS_CHANGED(cls, h3_data, **kwargs):
        return h3_data


NODE_CLASS_MAPPINGS = {"CSGlideH3DirectorCS": CSGlideH3Director}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideH3DirectorCS": "Glide H3 Director"}
