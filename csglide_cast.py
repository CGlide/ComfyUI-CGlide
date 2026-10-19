"""H3 Studio — one node covering both MiniMax H3 model families.

  mode "fl2va"  -> first / last keyframes      (needs minimax_h3_fl2va_* weights)
  mode "ref2va" -> 9 images / 3 videos / 3 audio (needs minimax_h3_ref2va_* weights)

Everything the UI holds lives in the single `h3_data` JSON widget. Reference tags
are written in the prompt as @image1 / @video1 / @audio1 and transcribed here to
the real <Picture i> / <Video k> / <Audio j> presentation, renumbered against the
slots that are actually filled.
"""

import asyncio
import json
import math
import os
import re

import numpy as np
from collections import deque
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

MIN_REF_FRAMES = 5          # H3 rejects a reference video shorter than this

# Candidate lengths for the smart span picker. Only 5 / 22 / 39 / 56 are worth
# offering: _refs() trims every reference video with `while n %% 17 != 5`, so a
# 17f pick silently collapses to 5 and a 27f pick collapses to 22. Anything not
# on that grid is a longer decode for the same conditioning.
SPAN_LADDER = (22, 39)
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


def snap_guide_run(n):
    """Largest 17k+5 run not exceeding n; 0 if there is not even a minimum run.

    MiniMaxH3AddGuide does this snap itself, but it then keeps the FIRST n
    frames of what it was handed. For a continuation the run has to END at the
    previous clip's last frame, so the snap happens here and the LAST n are
    kept instead.
    """
    n = int(n)
    if n < MIN_REF_FRAMES:
        return 0
    while n % 17 != 5:
        n -= 1
    return n


def flatten_exposure(frames, strength=1.0):
    """Level a guide run's brightness onto its own final frame.

    Measured on real renders: the model continues from the anchor window's
    AVERAGE exposure rather than from its last frame - consistent with the run
    being encoded as one block. A window ending below its average produced a
    clip about 19% brighter; one ending above produced a clip about 17% darker,
    and the sign followed the window both times. Scaling every frame onto the
    last one removes the cause rather than correcting the result, so nothing
    accumulates down a chain of clips.

    MEAN ONLY, deliberately: per-frame std matching was tried on the LTX chunk
    assembler and made exposure worse.

    The levelled frames are conditioning only. The model renders its own
    version of this span, so nothing here reaches the finished picture.
    """
    if strength <= 0.0 or frames.shape[0] < 2:
        return frames
    lum = frames.mean(dim=(1, 2, 3))
    target = float(lum[-1])
    if target <= 1e-6:
        return frames
    gain = 1.0 + (target / lum.clamp(min=1e-6) - 1.0) * float(strength)
    return (frames * gain.view(-1, 1, 1, 1)).clamp(0.0, 1.0)


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


def _clean_origin(raw):
    if not isinstance(raw, dict):
        return None
    name, sub = raw.get("name"), raw.get("subfolder") or ""
    if not isinstance(name, str) or not name or not isinstance(sub, str):
        return None
    kind = "temp" if raw.get("type") == "temp" else "output"
    return {"name": name[:260], "subfolder": sub[:260], "type": kind}


def _inside_dir(root, target):
    root = os.path.normcase(os.path.realpath(root))
    target = os.path.normcase(os.path.realpath(target))
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:
        return False


LATENT_SIDECAR = ".h3latent"
FULL_SIDECAR = ".h3full"


def _parse_upscale(raw):
    if not isinstance(raw, dict):
        return None
    try:
        ov = int(raw.get("overlap") or 0)
    except Exception:
        ov = 0
    return {"src": _clean_origin(raw.get("src")),
            "prev": _clean_origin(raw.get("prev")),
            "overlap": ov if ov in (5, 22, 39) else (22 if ov > 0 else 0)}


def _render_path(org, ext=None):
    """A recorded render ({name, subfolder, type}) -> an absolute path inside
    output/ or temp/, optionally with its extension swapped. None if it is
    not there or would land outside those folders."""
    if not org:
        return None
    try:
        root = (folder_paths.get_temp_directory() if org["type"] == "temp"
                else folder_paths.get_output_directory())
    except Exception:
        return None
    name = org["name"] if ext is None else os.path.splitext(org["name"])[0] + ext
    path = os.path.join(root, org["subfolder"], name)
    if not _inside_dir(root, path) or not os.path.isfile(path):
        return None
    return os.path.realpath(path)


def _find_latent_sidecar(cont):
    """The .h3latent Glide Video wrote beside the clip CONTINUE FROM holds.

    Looked up from the clip's ORIGIN (where the render was written, recorded
    when it was adopted) and, failing that, beside the adopted copy in input/.
    Every candidate is resolved with realpath and must stay inside output/,
    temp/ or input/ - the origin comes from the browser, so it is never
    trusted as a path.
    """
    cands = []
    org = cont.get("origin")
    if org:
        try:
            root = (folder_paths.get_temp_directory() if org["type"] == "temp"
                    else folder_paths.get_output_directory())
            path = os.path.join(root, org["subfolder"],
                                os.path.splitext(org["name"])[0] + LATENT_SIDECAR)
            cands.append((root, path))
        except Exception:
            pass
    local = _resolve_asset(cont.get("file"))
    if local:
        cands.append((folder_paths.get_input_directory(),
                      os.path.splitext(local)[0] + LATENT_SIDECAR))
    for root, path in cands:
        if _inside_dir(root, path) and os.path.isfile(path):
            return os.path.realpath(path)
    return None


def project_latents(picks, limit=512):
    """The .h3latent sidecars belonging to a project's renders.

    picks are the project's recorded renders ({name, subfolder, type}). A
    render in a subfolder of its own (the usual project/NN_name layout)
    claims EVERY sidecar in that folder, so superseded takes go too; one in
    the bare output root claims only its own sidecar, since that folder is
    shared with everything else. Every path is realpath-checked to stay
    inside output/ or temp/, and only *.h3latent files are ever returned.
    Returns {path: bytes}.
    """
    found = {}
    if not isinstance(picks, list):
        return found
    for raw in picks[:limit]:
        org = _clean_origin(raw)
        if not org:
            continue
        try:
            root = (folder_paths.get_temp_directory() if org["type"] == "temp"
                    else folder_paths.get_output_directory())
        except Exception:
            continue
        folder = os.path.realpath(os.path.join(root, org["subfolder"]))
        if not _inside_dir(root, folder) or not os.path.isdir(folder):
            continue
        if org["subfolder"].strip("/\\"):
            names = [n for n in os.listdir(folder) if n.endswith(LATENT_SIDECAR)]
        else:
            names = [os.path.splitext(os.path.basename(org["name"]))[0] + LATENT_SIDECAR]
        for n in names:
            f = os.path.realpath(os.path.join(folder, n))
            if (f.endswith(LATENT_SIDECAR) and _inside_dir(root, f)
                    and os.path.isfile(f)):
                try:
                    found[f] = os.path.getsize(f)
                except OSError:
                    pass
    return found


def _load_image(ref):
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Studio: image not found on disk: %s" % ref)
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


def _load_video_frames(ref, start, end, max_frames):
    """Decode [start, end) seconds and resample to FPS. Returns [N, H, W, 3].

    The frame count is derived from the REQUESTED span, not the decoded one:
    a short window can land between source frames and come back one short,
    which then trips the 5-frame minimum H3 needs. A little slack is decoded
    past the out point so the nearest-frame search always has something to
    reach for, and the result is padded if the source truly runs out.
    """
    if av is None:
        raise RuntimeError("H3 Studio: PyAV is required to read reference videos.")
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Studio: video not found on disk: %s" % ref)

    slack = 0.75                      # seconds decoded past the out point
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
            if end is not None and t > end + slack:
                break
            frames.append(frame.to_ndarray(format="rgb24"))
            times.append(t)
            if len(frames) > 4096:
                break

    if not frames:
        raise ValueError("H3 Studio: no frames decoded from %s in that trim range." % ref)

    if end is not None:
        want = int(round((end - start) * FPS)) + 1
    else:
        span = (times[-1] - times[0]) if len(times) > 1 else 0.0
        want = int(round(span * FPS)) + 1
    want = max(MIN_REF_FRAMES, min(want, max_frames))

    idx, cur = [], 0
    for k in range(want):
        target = start + k / FPS
        while cur + 1 < len(times) and abs(times[cur + 1] - target) <= abs(times[cur] - target):
            cur += 1
        idx.append(cur)

    if len(idx) < MIN_REF_FRAMES:     # source genuinely ran out - hold the last frame
        idx += [idx[-1]] * (MIN_REF_FRAMES - len(idx))

    arr = np.stack([frames[i] for i in idx]).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _load_video_tail(ref, n, start_hint=0.0):
    """Decode the LAST n frames of a clip, by POSITION rather than by time.

    The timestamp path samples targets at start + k/FPS, and start/end reach
    the node from the browser's video.duration - which for mkv is routinely a
    frame or more off the true last PTS. Read short, the window stops BEFORE
    the clip's final frame, and the continuation then picks up from slightly
    the wrong moment: the join reads as a small jump no colour or seam work
    can fix, because the two pieces are not adjacent in time.

    A continuation window is always the tail, so position is the honest
    measure. Only the last n frames are held, so the memory cost is the window
    and not the clip.
    """
    if av is None:
        raise RuntimeError("H3 Studio: PyAV is required to read reference videos.")
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Studio: video not found on disk: %s" % ref)

    keep = deque(maxlen=max(1, int(n)))
    total = 0
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base else 1.0 / FPS
        # a clip at another frame rate would give n frames of a different
        # duration, so leave those to the timestamp path
        rate = float(stream.average_rate) if stream.average_rate else FPS
        if abs(rate - FPS) > 0.51:
            return None
        seek_to = max(0.0, float(start_hint or 0.0) - 1.0)
        if seek_to > 0:
            try:
                container.seek(int(seek_to / tb), stream=stream)
            except Exception:
                pass
        for frame in container.decode(video=0):
            keep.append(frame.to_ndarray(format="rgb24"))
            total += 1
            if total > 8192:
                break

    if not keep:
        raise ValueError("H3 Studio: no frames decoded from %s." % ref)
    arr = np.stack(list(keep)).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _scan_video(path, short_edge=112, max_frames=4096):
    """Cheap per-frame stats for the whole clip: exposure, detail, motion.

    Decoded small and grey through libav's own scaler, and never held - only
    three scalars per frame survive, so a 300 frame clip costs a few kB and a
    fraction of a second. Everything the picker decides is decided from here.
    """
    if av is None:
        raise RuntimeError("H3 Studio: PyAV is required to read reference videos.")

    times, luma, detail, diff = [], [], [], []
    prev = None
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tb = float(stream.time_base) if stream.time_base else 1.0 / FPS
        for frame in container.decode(video=0):
            t = float(frame.pts * tb) if frame.pts is not None else (len(luma) / FPS)
            w = max(4, int(round(frame.width * short_edge / max(1, frame.height))))
            g = frame.reformat(width=w, height=short_edge,
                               format="gray").to_ndarray().astype(np.float32) / 255.0
            times.append(t)
            luma.append(float(g.mean()))
            # 4-neighbour Laplacian variance - high on real texture and edges,
            # near zero on flat fog, a blown frame or a motion-blurred smear
            lap = (g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1]
                   - 4.0 * g[1:-1, 1:-1])
            detail.append(float(lap.var()))
            diff.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
            prev = g
            if len(luma) >= max_frames:
                break

    if not luma:
        raise ValueError("H3 Studio: no frames decoded from %s." % path)
    return (np.asarray(times, dtype=np.float64), np.asarray(luma, dtype=np.float32),
            np.asarray(detail, dtype=np.float32), np.asarray(diff, dtype=np.float32))


def _cut_segments(diff, n_total):
    """Split the clip at hard cuts. Returns [(start, end), ...] index ranges.

    A window straddling a cut hands H3 two different framings as one reference,
    and because a confused reference falls back to the image slots, the look
    carry is lost with nothing in the log to say why. Cheap to avoid: a cut is
    a frame-to-frame difference far above the clip's own median.
    """
    body = diff[1:]
    med = float(np.median(body)) if body.size else 0.0
    thresh = max(0.05, med * 6.0)
    bounds = [0] + [i for i in range(1, n_total) if diff[i] > thresh] + [n_total]
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] >= MIN_REF_FRAMES]
    return (segs or [(0, n_total)]), thresh


def pick_reference_span(ref, ladder=SPAN_LADDER, prefer_tail=True):
    """Choose the span of a previous render that best SHOWS the scene.

    This is a reference, not a guide: it carries no timing, so unlike the
    CONTINUE FROM window it need not sit at the tail and need not be adjacent
    in time to anything. What it has to be is legible - correctly exposed, with
    real detail in it, and not spanning a cut.

    Returns {start, end, frames, score, note} in seconds, or None to leave the
    slot's own trim alone.
    """
    path = _resolve_asset(ref)
    if path is None:
        return None
    try:
        times, luma, detail, diff = _scan_video(path)
    except Exception as exc:
        print("[H3 Studio] span pick: could not scan %s (%s) - keeping the trim." % (ref, exc))
        return None

    n_total = len(luma)
    if n_total < MIN_REF_FRAMES:
        return None

    segments, thresh = _cut_segments(diff, n_total)
    detail_ref = float(np.percentile(detail, 90)) or 1.0

    # prefix sums so every candidate window is three subtractions
    cs_l = np.concatenate(([0.0], np.cumsum(luma, dtype=np.float64)))
    cs_d = np.concatenate(([0.0], np.cumsum(detail, dtype=np.float64)))
    cs_v = np.concatenate(([0.0], np.cumsum(diff, dtype=np.float64)))
    mean = lambda cs, s, e: (cs[e] - cs[s]) / max(1, e - s)

    best = None
    for si, (a, b) in enumerate(segments):
        is_last = (si == len(segments) - 1)
        for want in ladder:
            if b - a < want:
                continue
            for s in range(a, b - want + 1):
                e = s + want
                ml = mean(cs_l, s, e)
                if ml < 0.06 or ml > 0.94:          # black or blown - no information
                    continue
                expo = 1.0 - min(1.0, abs(ml - 0.45) / 0.45)
                det = min(1.0, mean(cs_d, s, e) / detail_ref)
                stab = 1.0 - min(1.0, mean(cs_v, s + 1, e) / thresh)
                score = 0.40 * det + 0.35 * expo + 0.25 * stab
                if want > ladder[0]:
                    score *= 1.03                   # ties go to the cheaper span
                if prefer_tail and is_last:
                    score *= 1.12                   # a preference, not a rule
                if best is None or score > best["score"]:
                    best = {"score": score, "s": s, "e": e, "want": want,
                            "seg": (a, b), "si": si + 1, "nseg": len(segments)}

    if best is None:
        print("[H3 Studio] span pick: nothing legible in %s - keeping the trim." % ref)
        return None

    s, want = best["s"], best["want"]
    start = float(times[s])
    end = start + (want - 1) / float(FPS)
    seg_len = best["seg"][1] - best["seg"][0]

    note = ""
    if best["score"] < 0.35:
        note = "weak"
    elif seg_len < 30:
        note = "short segment (%df)" % seg_len

    print("[H3 Studio] span pick: %s -> %.2fs..%.2fs (%df, score %.2f, "
          "segment %d/%d)%s"
          % (ref, start, end, want, best["score"], best["si"], best["nseg"],
             (" - " + note) if note else ""))

    return {"start": start, "end": end, "frames": want,
            "score": best["score"], "note": note}


def _load_audio(ref, start, end):
    """Returns a ComfyUI AUDIO dict, trimmed to [start, end)."""
    path = _resolve_asset(ref)
    if path is None:
        raise FileNotFoundError("H3 Studio: audio not found on disk: %s" % ref)

    waveform, sr = None, None
    if torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            waveform = None
    if waveform is None:
        if av is None:
            raise RuntimeError("H3 Studio: cannot read audio (no torchaudio, no PyAV).")
        chunks = []
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError("H3 Studio: %s has no audio stream." % ref)
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
        raise ValueError("H3 Studio: audio trim for %s is empty." % ref)
    waveform = waveform[..., a:b]
    # H3 packs exactly two audio channels. pack_audio() emits channels * T rows
    # while PackedLayout reserves T * 2, so anything but stereo makes the two
    # disagree and the mismatch only surfaces inside the sampler.
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2].contiguous()
    return {"waveform": waveform[None, ...], "sample_rate": int(sr)}


def _load_video_soundtrack(ref, start, end):
    try:
        return _load_audio(ref, start, end)
    except Exception as e:
        print("[H3 Studio] no usable soundtrack in %s (%s) — video sent silent." % (ref, e))
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


def _video_slot_list(raw):
    """Video slots, whitelisted the same way `cont` is.

    `carry` and `pick` are the batch-mode keys: carry means this slot holds the
    previous render as a LOOK reference (and earns the injected job line), pick
    means let pick_reference_span() choose the window instead of the UI trim.
    They are independent - a hand-trimmed carry slot is a legitimate thing.
    """
    out = []
    src = raw if isinstance(raw, list) else []
    for i in range(MAX_VIDEOS):
        item = src[i] if i < len(src) and isinstance(src[i], dict) else {}
        if not item.get("file"):
            out.append(None)
            continue
        end = item.get("end")
        out.append({
            "file": str(item["file"]),
            "start": float(item.get("start") or 0.0),
            "end": float(end) if end not in (None, "") else None,
            "audio": bool(item.get("audio")),
            "carry": bool(item.get("carry")),
            "pick": bool(item.get("pick")),
        })
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

    # CONTINUE FROM: the previous clip's tail, anchored at frame 0 of this one.
    # Whitelisted like everything else - an older project file cannot smuggle
    # anything in through it.
    raw_cont = data.get("cont")
    cont = None
    if isinstance(raw_cont, dict) and raw_cont.get("file"):
        end = raw_cont.get("end")
        start = float(raw_cont.get("start") or 0.0)
        end = float(end) if end not in (None, "") else None
        # The UI states the window as start/end, and the span IS the frame
        # count - reading a separate "frames" field instead would silently cap
        # a 39f pick back to whatever that field last said.
        if end is not None:
            frames = int(round((end - start) * FPS)) + 1
        else:
            frames = int(raw_cont.get("frames") or 22)
        cont = {
            "file": str(raw_cont["file"]),
            "start": start,
            "end": end,
            "frames": frames,
            "audio": bool(raw_cont.get("audio")),
            # the UI sends a checkbox; a float still works if a file carries one
            "flatten": max(0.0, min(1.0, float(raw_cont.get("flatten") or 0.0))),
            # Where the adopted clip came from ({name, subfolder, type} in
            # output/ or temp/). Only used to find its latent sidecar, and
            # only through _resolve_render, which keeps it inside those folders.
            "origin": _clean_origin(raw_cont.get("origin")),
        }

    # Who this clip IS. Whitelisted like the rest, so a project file that
    # predates this simply has no "clip" key and the node emits an empty name.
    raw_clip = data.get("clip")
    clip = None
    if isinstance(raw_clip, dict):
        try:
            idx = int(raw_clip.get("index") or 0)
            cnt = int(raw_clip.get("count") or 0)
        except Exception:
            idx = cnt = 0
        clip = {
            "name": str(raw_clip.get("name") or "")[:120],
            "index": max(0, idx),
            "count": max(0, cnt),
            "project": str(raw_clip.get("project") or "")[:120],
        }

    out = {
        "mode": "fl2va" if data.get("mode") == "fl2va" else "ref2va",
        "width": int(data.get("width") or 1344),
        "height": int(data.get("height") or 768),
        "length": int(data.get("length") or 124),
        "ref_image_size": "max" if data.get("ref_image_size") == "max" else "match",
        # Multiplier for the refine pass's references ONLY. 1.0 = today's
        # behaviour and nothing extra is encoded. See _refs.
        "ref_refine_scale": max(1.0, min(4.0, float(data.get("ref_refine_scale") or 1.0))),
        "prompt": str(data.get("prompt") or ""),
        "first": one("first"),
        "last": one("last"),
        "images": _slot_list(slots.get("images"), MAX_IMAGES),
        "videos": _video_slot_list(slots.get("videos")),
        "audios": _slot_list(slots.get("audios"), MAX_AUDIOS),
        "cont": cont,
        # LATENT HOLD: continue from the previous clip's sampled latent instead
        # of a re-encoded guide. A node setting, sent at the top of h3_data
        # because Render all clears and refills `cont` on every link.
        "cont_latent": bool(data.get("cont_latent")),
        # UPSCALE PASS: build this clip from its saved full latent (.h3full,
        # next to its draft render) instead of an empty one, and hand Glide
        # Join the previous clip's UPSCALED render. Sent by the panel only
        # while the Render row's "upscale" box is on.
        "upscale": _parse_upscale(data.get("upscale")),
        "clip": clip,
        "song": _parse_song(data.get("song")),
        "audio_mode": "timeline" if data.get("audio_mode") == "timeline" else "ref",
    }
    # TIMELINE AUDIO. The audio slots either feed the encoder as references
    # (<Audio j>, the default) or are PLACED: written into the audio half of the
    # target latent at a time in the clip and held there, with H3 generating its
    # own sound in the gaps. fl2va has no audio references at all, so there the
    # slots are always placed. A placed slot is moved out of "audios" so the
    # reference path and the tag map never see it.
    if out["mode"] == "fl2va" or out["audio_mode"] == "timeline":
        out["placed"] = _placed_list(slots.get("audios"))
        out["audios"] = [None] * MAX_AUDIOS
    else:
        out["placed"] = [None] * MAX_AUDIOS
    return out


def _placed_list(raw):
    """Audio slots as timeline pieces: which part of the file, where it lands."""
    out = []
    src = raw if isinstance(raw, list) else []

    def num(v, d):
        try:
            v = float(v)
        except Exception:
            return d
        return v if math.isfinite(v) else d

    for i in range(MAX_AUDIOS):
        item = src[i] if i < len(src) and isinstance(src[i], dict) else {}
        if not item.get("file"):
            out.append(None)
            continue
        start = max(0.0, num(item.get("start"), 0.0))
        end = item.get("end")
        end = num(end, None) if end not in (None, "") else None
        if end is not None and end <= start:
            end = None
        out.append({
            "file": str(item["file"]),
            "start": start,
            "end": end,
            "at": max(0.0, num(item.get("at"), 0.0)),
            "hold": max(0.0, min(1.0, num(item.get("hold"), 1.0))),
        })
    return out


def _parse_song(raw):
    """SONG: the project's track, already windowed for THIS clip by the UI.

    The panel knows the whole project - every clip's length and which ones are
    linked - so it works out where this clip sits in the song and sends only
    that: the file, the second this clip's frame 0 lands on, and how hard to
    hold it. Nothing here needs the other clips.
    """
    if not isinstance(raw, dict) or not raw.get("file"):
        return None
    try:
        start = float(raw.get("start") or 0.0)
    except Exception:
        start = 0.0
    try:
        strength = float(raw.get("strength"))
    except Exception:
        strength = 1.0
    if not math.isfinite(start):
        start = 0.0
    if not math.isfinite(strength):
        strength = 1.0
    return {
        "file": str(raw["file"]),
        "start": max(0.0, start),
        "strength": max(0.0, min(1.0, strength)),
    }


# Clip labels are free text typed into the panel, and they end up in a path.
# Windows rejects more than POSIX does, so this clears the union: the reserved
# characters, control codes, the device names that stay illegal even with an
# extension, and trailing dots or spaces (Explorer strips those silently, so a
# file written with one cannot be reopened under the name it was given).
_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    "%s%d" % (p, i) for p in ("COM", "LPT") for i in range(1, 10)}

_ILLEGAL = re.compile(r'[\x00-\x1f<>:"/\\|?*]+')


def safe_name(text, limit=60):
    """Turn a typed clip label into something usable as a filename fragment."""
    s = _ILLEGAL.sub("-", str(text or "").strip())
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[-_.]{2,}", "-", s).strip("-_. ")[:limit].strip("-_. ")
    # Last, not first: the escaping prefix must survive the trim that follows,
    # and truncation can expose a reserved name that was not one at full length.
    if s.upper().split(".")[0] in _WIN_RESERVED:
        s = "_" + s
    return s


def clip_label(cfg):
    """A ready-made filename_prefix for this clip: <project>/<NN>_<name>.

    Shaped to be wired straight into Glide Video's filename_prefix, so the
    project becomes a subfolder and each render arrives already named. That is
    why the project is a path segment rather than part of the stem: one folder
    per film beats one folder holding every film's clips, and the prefix
    widget is the one place in the graph that already understands a slash.

    Index leads the name because that is what makes the folder sort into the
    order of the film. A name alone puts "Twirls" before "Opening Steps",
    which is the same problem in a nicer font. Padding follows the project
    size, so a twelve clip film reads 01..12 and never 1, 10, 11, 2.

    The panel's placeholder is not a name: if nobody typed one the clip is
    called "Clip 4" only because it is fourth, so the index already says it.
    """
    clip = cfg.get("clip")
    if not clip:
        return ""
    idx, cnt = clip["index"], clip["count"]
    name = safe_name(clip["name"])
    project = safe_name(clip["project"])

    if not idx:
        stem = name
    else:
        width = max(2, len(str(max(cnt, idx))))
        if not name or clip["name"].strip().lower() == ("clip %d" % idx):
            stem = "%0*d" % (width, idx)
        else:
            stem = "%0*d_%s" % (width, idx, name)

    if project and stem:
        return "%s/%s" % (project, stem)
    return project or stem


CARRY_NOTE = ("@video%d gives the state of the scene at this point in the film: "
              "follow its lighting, atmosphere, colour grade and set dressing. "
              "Do not follow its camera movement or its action.")


def inject_carry_note(cfg, tags=None):
    """Prepend the job line for any slot the batch mode filled as a look carry.

    Injected as an @token so the normal tag machinery renumbers it against the
    filled slots - writing <Video k> here would go stale the moment a slot above
    it emptied. A slot already mentioned by hand is left alone: the prompt wins
    over the mode. "By hand" covers both forms - the @video token, and the raw
    <Video k> tag the slot is emitted as (official-guide prompts write those
    directly, and transcription passes them through untouched).
    """
    prompt = cfg["prompt"]
    tags = tags or {}
    lines = []
    for n, slot in enumerate(cfg["videos"], start=1):
        if slot is None or not slot.get("carry"):
            continue
        if re.search(r"@video%d(?![0-9A-Za-z])" % n, prompt):
            continue
        real = tags.get("@video%d" % n)
        if real and real in prompt:
            continue
        lines.append(CARRY_NOTE % n)
    if not lines:
        return prompt
    return "\n".join(lines) + "\n\n" + prompt


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


def strip_placed_tokens(prompt, cfg):
    """A timeline slot is not a reference: the encoder never sees it, so its
    @audio token has nothing to point at. Dropped here with its own reason,
    rather than reaching transcription and being reported as an empty slot."""
    out = prompt
    for n, slot in enumerate(cfg.get("placed") or [], start=1):
        tok = "@audio%d" % n
        if slot and re.search(re.escape(tok) + r"(?![0-9A-Za-z])", out):
            print("[H3 Studio] %s is on the timeline, not a reference - dropped from "
                  "the prompt (describe the sound in words instead)." % tok)
            out = re.sub(re.escape(tok) + r"(?![0-9A-Za-z])", "", out)
    return out


def transcribe_prompt(prompt, tags, known_tokens):
    """Replace @tokens with real tags. Tokens pointing at empty slots are dropped."""
    out = prompt
    for token in sorted(known_tokens, key=len, reverse=True):
        if token in tags:
            out = out.replace(token, tags[token])
        elif token in out:
            print("[H3 Studio] %s points at an empty slot — removed from the prompt." % token)
            out = out.replace(token, "")
    return out


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

_SPAN_CACHE = {}


def pick_reference_span_cached(ref, ladder=SPAN_LADDER, prefer_tail=True):
    """pick_reference_span keyed on path + mtime + size.

    The UI asks for a pick when the slot is filled and the node asks again at
    build time; without this the clip is scanned twice for the same answer.
    Keyed on the file's own identity, so a re-render to the same name still
    rescans.
    """
    path = _resolve_asset(ref)
    if path is None:
        return None
    try:
        stat = os.stat(path)
        key = (path, int(stat.st_mtime), int(stat.st_size), tuple(ladder), bool(prefer_tail))
    except Exception:
        return pick_reference_span(ref, ladder, prefer_tail)
    if key in _SPAN_CACHE:
        return _SPAN_CACHE[key]
    got = pick_reference_span(ref, ladder, prefer_tail)
    if len(_SPAN_CACHE) > 64:
        _SPAN_CACHE.clear()
    _SPAN_CACHE[key] = got
    return got


# The UI needs the same answer the build will use, so the pick lives here rather
# than being re-implemented in the browser. Registered defensively: if the server
# module moves, the node still loads and the picker still runs at build time.
try:
    from aiohttp import web as _web
    from server import PromptServer as _PromptServer

    @_PromptServer.instance.routes.post("/cglide/latents")
    async def _latents(request):
        """Scan (default) or delete a project's latent sidecars. The browser
        sends the project's recorded renders; what is touched is decided here
        by project_latents(), never by a path from the request."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        picks = body.get("picks") if isinstance(body, dict) else None
        found = project_latents(picks)
        deleted, failed = 0, 0
        if isinstance(body, dict) and body.get("delete") is True:
            for f in list(found):
                try:
                    os.remove(f)
                    deleted += 1
                except OSError as e:
                    failed += 1
                    print("[H3 Studio] could not delete %s (%s)" % (f, e))
            print("[H3 Studio] deleted %d latent sidecar%s (%.1f MB)%s"
                  % (deleted, "" if deleted == 1 else "s",
                     sum(found.values()) / 1048576.0,
                     (", %d failed" % failed) if failed else ""))
        return _web.json_response({"count": len(found), "bytes": sum(found.values()),
                                   "deleted": deleted, "failed": failed})

    @_PromptServer.instance.routes.get("/cglide/pick_span")
    async def _pick_span(request):
        ref = request.rel_url.query.get("file") or ""
        if not ref:
            return _web.json_response({"error": "no file"}, status=400)
        try:
            # PyAV decoding is blocking, and this handler runs ON ComfyUI's own
            # event loop - scanning inline would freeze the server, the progress
            # websocket included, for as long as the decode takes. On a long clip
            # that is indistinguishable from a hung render.
            loop = asyncio.get_running_loop()
            got = await loop.run_in_executor(None, pick_reference_span_cached, ref)
        except Exception as exc:
            return _web.json_response({"error": str(exc)}, status=500)
        if got is None:
            return _web.json_response({"ok": False})
        return _web.json_response({
            "ok": True,
            "start": round(float(got["start"]), 4),
            "end": round(float(got["end"]), 4),
            "frames": int(got["frames"]),
            "score": round(float(got["score"]), 3),
            "note": got["note"],
        })
except Exception as _exc:  # pragma: no cover
    print("[H3 Studio] /cglide/pick_span not registered (%s)" % _exc)


class CSGlideCast:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                # NOT multiline: the v2 frontend renders a multiline STRING as a
                # real textarea overlay, which floats over the node's own UI and
                # eats every click in that band. A single-line widget is drawn on
                # the canvas and can be hidden properly.
                "h3_data": ("STRING", {"default": ""}),
            },
            "optional": {
                "audio_vae": ("VAE",),
                "first_frame": ("IMAGE",),
                "last_frame": ("IMAGE",),
            },
        }

    # Reordered and trimmed in 1.4.5 (his call, saved workflows rewired once):
    # guide_frames is gone; the two conditionings sit together. 1.4.6 brought
    # width / height / length / seconds back (people DO drive their whole
    # workflow from H3 Studio, issue from MADmotionAI) plus refine_multiplier,
    # all APPENDED after positive_refine so 1.4.5 workflows keep their wires.
    # LiteGraph stores links by output INDEX - from here on, append, never
    # insert, or every saved workflow is silently rewired again.
    #   overlap_frames / source_video / locked_audio -> Glide Join (and Glide
    #   Upscale Hold); locked_audio is the held song/timeline sound as raw
    #   samples with a splice weight, empty when nothing is held.
    RETURN_TYPES = ("LATENT", "INT", "STRING", "STRING", "AUDIO",
                    "CONDITIONING", "CONDITIONING",
                    "INT", "INT", "INT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("latent", "overlap_frames", "source_video", "clip_name",
                    "locked_audio", "positive", "positive_refine",
                    "width", "height", "length", "seconds", "refine_multiplier")
    FUNCTION = "build"
    CATEGORY = "CGlide"
    DESCRIPTION = "MiniMax H3 director — first/last keyframes or omni references, with automatic reference tagging."

    # ---------------- conditioning variants ----------------

    @staticmethod
    def _without_keyframes(cond):
        """The same conditioning minus the keyframe block.

        A keyframe is encoded at the canvas, and the model reserves target rows
        for it from the TARGET latent's shape: `all_video_rows[~img_update] =
        cond_video_rows`. Feed the same conditioning to a second sampler running
        on an upscaled latent and those two disagree - 1344x768 gives one row
        count, 1920x1088 another - and the assignment raises a broadcast error
        before the first step. That is what a latent upscaler between two
        samplers does: same guider, bigger latent.

        References are NOT stripped. They carry their own spatial shape and the
        layout allocates from that, so they survive a resolution change - a
        clip with references and no continuation refines fine today, which is
        the evidence for leaving them alone.

        Dropping the anchor for a refine pass costs nothing real: the guide's
        job is to place the continuation at the seam, and that is decided in the
        first pass. A low-sigma refine is detail work on frames already
        committed.
        """
        out = []
        for item in cond:
            meta = dict(item[1]) if len(item) > 1 and isinstance(item[1], dict) else {}
            meta.pop("minimax_keyframes", None)
            meta.pop("minimax_frame_count", None)
            out.append([item[0], meta])
        return out

    @staticmethod
    def _audit_audio(cond, latent=None):
        """Check the audio row budget here, where the blocks still have names.

        PackedLayout reserves cond-audio rows from `ref_audio_t` (references) and
        from `audio_latent.shape[-1]` (keyframes), then fills them with
        `pack_audio()`, which emits `channels * T` rows. Two ways those can
        disagree: a declared length that is not the tensor's length, or a latent
        that is not stereo. Either one surfaces inside the sampler as
        `all_audio_rows[~audio_update] = cond_audio_rows` failing to broadcast,
        with nothing in the message saying which slot did it.

        The order matters as much as the totals: the layout packs keyframe audio
        first, then reference audio in list order, and model_base builds
        cond_audio_latents the same way. This walks them in that same order so
        the printed line reads as the packed sequence does.
        """
        meta = {}
        if cond and len(cond[0]) > 1 and isinstance(cond[0][1], dict):
            meta = cond[0][1]

        # How many audio latent frames this shot actually has room for. Every
        # reference frame past it is conditioning the model with audio the clip
        # cannot use.
        grid = 0
        try:
            grid = int(latent["samples"].tensors[1].shape[-1])
        except Exception:
            grid = 0

        reserved = filled = 0
        report = []
        overlong = []

        def account(label, declared, latent):
            ch = int(latent.shape[2])
            t = int(latent.shape[-1])
            if ch != 2:
                raise ValueError(
                    "H3 Studio: %s encoded to %d audio channel(s); H3 packs stereo. "
                    "Re-export that file as stereo, or leave its sound off."
                    % (label, ch))
            if declared != t:
                raise ValueError(
                    "H3 Studio: %s declares %d audio latent frames but holds %d. "
                    "The model would reserve %d rows and get %d."
                    % (label, declared, t, declared * 2, ch * t))
            report.append("%s %d" % (label, t))
            return declared * 2, ch * t

        for i, kf in enumerate(meta.get("minimax_keyframes") or []):
            al = kf.get("audio_latent")
            if al is None:
                continue
            label = ("continuation guide" if kf.get("resolved_frame_index") == 0
                     else "keyframe %d" % (i + 1))
            r, f = account(label, int(al.shape[-1]), al)
            reserved += r
            filled += f

        for i, blk in enumerate(meta.get("minimax_refs") or []):
            al = blk.get("audio_latent")
            if al is None or not blk.get("ref_audio_t"):
                continue
            label = "%s ref %d" % (blk.get("kind", "?"), i + 1)
            r, f = account(label, int(blk["ref_audio_t"]), al)
            reserved += r
            filled += f
            # A reference several times longer than the shot has been linked to
            # a sampler crash (issue #8: ~51s of reference audio against a 9.4s
            # clip). The threshold is not a measured limit -- it sits between a
            # reference that worked at 1.3x the grid and one that crashed at
            # 5.4x -- so this warns and renders rather than refusing.
            if grid and int(al.shape[-1]) > grid * 3:
                overlong.append((label, int(al.shape[-1])))

        if report:
            print("[H3 Studio] audio latent frames: " + ", ".join(report)
                  + "  ->  %d cond rows" % filled)
        for label, t in overlong:
            print("[H3 Studio] WARNING: %s is %.1fs of audio against a %.1fs "
                  "clip. Only the first %.1fs can be used, and references this "
                  "long have been linked to a sampler crash. Trim it closer to "
                  "the clip length."
                  % (label, t / float(AUDIO_LATENT_FPS),
                     grid / float(AUDIO_LATENT_FPS),
                     grid / float(AUDIO_LATENT_FPS)))
        if reserved != filled:
            raise ValueError(
                "H3 Studio: audio conditioning would reserve %d rows and fill %d. "
                "Report this line with the render settings." % (reserved, filled))

    # ---------------- reference encoding ----------------

    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        if audio_vae is None:
            raise ValueError("H3 Studio: audio references need the audio VAE connected "
                             "(minimax_h3_audio_vae_fp32).")
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            if torchaudio is None:
                raise RuntimeError("H3 Studio: torchaudio is required to resample reference audio.")
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z, z.shape[-1]

    def _refs(self, cfg, vae, audio_vae, width, height, frame_count, scale_mul=1.0):
        """scale_mul multiplies the size the IMAGE references are encoded at.

        The refine pass samples an upscaled latent, but its references were
        sized for the base canvas - at 0.5 MP base into a 1.5 MP upscale they
        are a third of the resolution of the thing being sharpened. Encoding a
        second, larger set for that pass closes the gap, and the wider the gap
        the more it is worth: at his own 1.03 -> 1.5 MP it is a small lift, at
        3x it is not.

        Images only, on purpose. A video reference is frame_count frames, so
        scaling it multiplies VRAM by the length of the clip for detail that
        motion conditioning does not carry anyway.
        """
        ref_items, ref_blocks = [], []
        size_mode = cfg["ref_image_size"]
        scale_mul = max(1.0, float(scale_mul or 1.0))

        for slot in cfg["images"]:
            if slot is None:
                continue
            img = _load_image(slot["file"])
            h, w = img.shape[1], img.shape[2]
            if size_mode == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            # Still capped at 1.0: the multiplier lifts a reference back toward
            # its own pixels, it never invents any. A 4000px source scaled to
            # 0.3 for the canvas has plenty of room; a 512px one does not, and
            # stretching that would only cost VRAM.
            scale = min(1.0, scale * scale_mul)
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
            if slot.get("pick"):
                picked = pick_reference_span_cached(slot["file"])
                if picked is not None:
                    start, end = picked["start"], picked["end"]
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
            if n < MIN_REF_FRAMES:
                raise ValueError(
                    "H3 Studio: %s gave only %d frames after trimming; H3 needs %d. "
                    "Widen the trim, or use the 'last 22f' quick pick."
                    % (slot["file"], n, MIN_REF_FRAMES))
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

    # ---------------- continuation ----------------

    # ---------------- locked audio: song + timeline pieces ----------------

    @staticmethod
    def _silence():
        return {"waveform": torch.zeros((1, 2, 0)), "sample_rate": 32000}

    # Inner edges of a placed piece hand over to H3 over this many audio latent
    # frames (40 per second), so the held sound and the generated sound meet in
    # a blend instead of a hard butt joint. Clip edges and edges against the
    # song need no handover and get none.
    EDGE_FRAMES = 4

    def _lock_audio(self, cfg, audio_vae, latent, seconds):
        """Write held audio into the AUDIO half of the target latent and pin it.

        ComfyUI's MiniMaxH3 model takes a per-stream noise mask, per audio
        latent frame: a row at mask 0 runs at AUDIO_COND_TIMESTEP and is
        re-injected clean every step (model_base.MiniMaxH3.scale_latent_inpaint),
        a row at mask 1 is generated as usual, anything between is held that
        much. So sound can be held over part of a clip and generated over the
        rest - the same inpainting the LTX director does, on H3's audio.

        Two layers, song first:
          song   - the project track, over the whole clip, at its own hold.
          placed - audio slots on the timeline, each a [start, end] window of
                   its file landing at `at` seconds into the clip. A later slot
                   is written over an earlier one where they overlap.

        Returns (latent, locked) where locked is the held sound as raw samples,
        untouched by the VAE, plus a per-sample weight under "cglide_weight":
        1 where the original should replace the decoded sound outright, 0 where
        H3's own sound stands, ramped across the handovers. Glide Join splices
        with it. Only a hold of exactly 1 is spliced - below that the point is
        to let H3 rework the sound, so the decoded version is the one to keep.
        """
        song = cfg.get("song")
        placed = [p for p in (cfg.get("placed") or []) if p]
        if not song and not placed:
            return latent, self._silence()
        if audio_vae is None:
            raise ValueError("H3 Studio: held audio (the song or timeline audio) needs "
                             "the audio VAE connected (minimax_h3_audio_vae_fp32).")

        video, audio = latent["samples"].unbind()
        T = int(audio.shape[-1])
        new_audio = audio.clone()
        mask = torch.ones(T, dtype=torch.float32)      # 1 = generate
        rate = None
        layers = []                                     # (wave, sr, frame0, frames, hold, ramp_in, ramp_out)

        def load(file, start, end):
            try:
                return _load_audio(file, start, end)
            except ValueError:
                return None                             # window past the end of the file

        if song:
            start = song["start"]
            tr = load(song["file"], start, start + seconds)
            sr = int(tr["sample_rate"]) if tr else 32000
            wf = tr["waveform"] if tr else torch.zeros((1, 2, 0))
            need = int(round(seconds * sr))
            short = need - int(wf.shape[-1])
            if short > 0:
                wf = torch.cat([wf, torch.zeros(wf.shape[:-1] + (short,), dtype=wf.dtype)], dim=-1)
            wf = wf[..., :need]
            rate = sr
            layers.append((wf, sr, 0, T, song["strength"], False, False))
            print("[H3 Studio] song: %s %.3fs..%.3fs held at %.2f%s"
                  % (song["file"], start, start + seconds, song["strength"],
                     "" if short <= 0 else
                     " - the song ends %.2fs early, silence after" % (short / float(sr))))

        for n, p in enumerate(cfg.get("placed") or [], start=1):
            if not p:
                continue
            f0 = int(round(p["at"] * AUDIO_LATENT_FPS))
            if f0 >= T:
                print("[H3 Studio] audio %d: placed at %.2fs, past the end of this "
                      "%.2fs clip - skipped" % (n, p["at"], seconds))
                continue
            room = (T - f0) / float(AUDIO_LATENT_FPS)
            end = p["end"]
            want_end = p["start"] + room if end is None else min(end, p["start"] + room)
            tr = load(p["file"], p["start"], want_end)
            if tr is None:
                print("[H3 Studio] audio %d: nothing in %s at %.2fs - skipped"
                      % (n, p["file"], p["start"]))
                continue
            sr = int(tr["sample_rate"])
            wf = tr["waveform"]
            frames = min(T - f0, int(round(wf.shape[-1] / float(sr) * AUDIO_LATENT_FPS)))
            if frames <= 0:
                continue
            if rate is None:
                rate = sr
            layers.append((wf, sr, f0, frames, p["hold"],
                           f0 > 0 and not song, f0 + frames < T and not song))
            print("[H3 Studio] audio %d: %s %.2fs..%.2fs placed at %.2fs..%.2fs in the "
                  "clip, held at %.2f" % (n, p["file"], p["start"],
                                          p["start"] + frames / float(AUDIO_LATENT_FPS),
                                          f0 / float(AUDIO_LATENT_FPS),
                                          (f0 + frames) / float(AUDIO_LATENT_FPS), p["hold"]))

        if not layers:
            return latent, self._silence()

        rate = rate or 32000
        n_out = int(round(seconds * rate))
        exact = torch.zeros((2, n_out))
        weight = torch.zeros(n_out)
        per_frame = rate / float(AUDIO_LATENT_FPS)

        for wf, sr, f0, frames, hold, r_in, r_out in layers:
            # latent: encode this piece on its own, starting on its own frame
            z, _ = self._encode_ref_audio(audio_vae, {"waveform": wf, "sample_rate": sr})
            z = z.to(device=new_audio.device, dtype=new_audio.dtype)
            k = min(frames, int(z.shape[-1]))
            new_audio[..., f0:f0 + k] = z[..., :k].expand(new_audio.shape[0], -1, -1, -1)

            # how much of each frame is held: 1 inside, ramped at inner edges
            held = torch.ones(k)
            e = min(self.EDGE_FRAMES, k // 2)
            if e > 0:
                ramp = torch.arange(1, e + 1, dtype=torch.float32) / (e + 1)
                if r_in:
                    held[:e] = ramp
                if r_out:
                    held[k - e:] = ramp.flip(0)
            mask[f0:f0 + k] = 1.0 - hold * held

            # exact samples for the splice, at the output rate
            w = wf[0] if wf.dim() == 3 else wf
            if sr != rate:
                if torchaudio is None:
                    raise RuntimeError("H3 Studio: torchaudio is needed to mix audio "
                                       "files at different sample rates.")
                w = torchaudio.functional.resample(w, sr, rate)
            s0 = int(round(f0 * per_frame))
            s1 = min(n_out, int(round((f0 + k) * per_frame)))
            m = max(0, min(s1 - s0, int(w.shape[-1])))
            exact[:, s0:s0 + m] = w[:2, :m]
            if hold >= 1.0 - 1e-6:
                ws = torch.nn.functional.interpolate(held.view(1, 1, -1), size=m,
                                                     mode="linear", align_corners=True).view(-1)
                weight[s0:s0 + m] = ws
            else:
                weight[s0:s0 + m] = 0.0

        a_mask = mask.view(1, 1, 1, T).expand_as(new_audio).to(new_audio.dtype).clone()
        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((video, new_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((torch.ones_like(video), a_mask))
        held_frames = int((mask < 1.0 - 1e-6).sum())
        if held_frames >= T:
            print("[H3 Studio] audio held over the whole %.2fs clip" % seconds)
        else:
            print("[H3 Studio] audio held over %.2fs of %.2fs, H3 generates the rest"
                  % (held_frames / float(AUDIO_LATENT_FPS), seconds))
        return out, {"waveform": exact[None].contiguous(), "sample_rate": int(rate),
                     "cglide_weight": weight}

    @staticmethod
    def _load_full(up, cfg):
        """The clip's saved full latent, as (latent, frame_count, width, height)."""
        path = _render_path(up.get("src"), FULL_SIDECAR)
        if path is None:
            who = clip_label(cfg) or "this clip"
            raise FileNotFoundError(
                "H3 Studio upscale pass: %s has no full latent (.h3full) next to its "
                "draft render. Render its draft with save_latent = full on Glide "
                "Video, then run the upscale pass again." % who)
        from safetensors.torch import load_file
        saved = load_file(path)
        video = saved.get("video")
        if video is None or video.dim() != 5:
            raise ValueError("H3 Studio upscale pass: %s holds no video latent"
                             % os.path.basename(path))
        dev = comfy.model_management.intermediate_device()
        video = video.to(dev)
        audio = saved.get("audio")
        if audio is not None:
            samples = comfy.nested_tensor.NestedTensor((video, audio.to(dev)))
        else:
            samples = video
        t = int(video.shape[2])
        frames = 5 if t <= 2 else ((t - 2) // 5) * 17 + 5
        print("[H3 Studio] upscale pass: loaded %s" % os.path.basename(path))
        return ({"samples": samples}, frames,
                int(video.shape[4]) * 16, int(video.shape[3]) * 16)

    def _latent_hold(self, cfg, vae, latent, frame_count):
        """LATENT HOLD: put the previous clip's sampled tail in the head of
        this clip's latent and pin it with the noise mask.

        Same machinery as the song lock, on the video half: held rows are
        re-injected clean every step, the rest is generated. The previous
        clip's frames are never decoded and re-encoded on the way in, so the
        seam carries no VAE round trip, no chroma loss, no 8-bit rounding.

        H3's video latent is laid out as a 2-frame head (5 pictures) followed
        by 5-frame blocks (17 pictures each), so the hold takes the last
        2 + 5k latent frames - the same count a 22 / 39 frame guide would
        encode to. Whether a tail slice behaves as a clean head is the thing
        to verify on the first render: guide_frames shows the held span
        decoded, and it should look like the end of the previous clip.

        Returns (latent, overlap_frames, preview) or None to fall back to the
        guide path.
        """
        cont = cfg.get("cont")
        if not cont:
            return None
        side = _find_latent_sidecar(cont)
        if side is None:
            print("[H3 Studio] latent hold: no .h3latent beside %s - wire the "
                  "first sampler's LATENT into Glide Video. Falling back to the "
                  "guide." % cont["file"])
            return None
        try:
            from safetensors.torch import load_file
            saved = load_file(side)
        except Exception as e:
            print("[H3 Studio] latent hold: could not read %s (%s) - falling back "
                  "to the guide." % (side, e))
            return None

        video, audio = latent["samples"].unbind()
        pv = saved.get("video")
        if pv is None or pv.dim() != 5:
            print("[H3 Studio] latent hold: %s holds no video latent - falling "
                  "back to the guide." % os.path.basename(side))
            return None
        if tuple(pv.shape[3:]) != tuple(video.shape[3:]) or pv.shape[1] != video.shape[1]:
            print("[H3 Studio] latent hold: sidecar is %dx%d latent, this clip is "
                  "%dx%d - render both at the same canvas, and save the FIRST "
                  "sampler's latent, not the refine pass. Falling back to the guide."
                  % (pv.shape[4], pv.shape[3], video.shape[4], video.shape[3]))
            return None

        n = snap_guide_run(cont["frames"])
        if n < MIN_REF_FRAMES:
            n = 22
        k = video_latent_t(n)
        if k >= video.shape[2] or k > pv.shape[2]:
            print("[H3 Studio] latent hold: %d latent frames do not fit (previous "
                  "%d, this clip %d) - falling back to the guide."
                  % (k, pv.shape[2], video.shape[2]))
            return None

        new_video = video.clone()
        new_video[:, :, :k] = pv[:, :, -k:].to(device=video.device, dtype=video.dtype) \
            .expand(video.shape[0], -1, -1, -1, -1)
        v_mask = torch.ones_like(video)
        v_mask[:, :, :k] = 0.0

        # audio: keep any lock already on it (the song), otherwise hold the
        # previous clip's own sound over the same span when its sound is on
        old_mask = latent.get("noise_mask")
        if old_mask is not None:
            a_mask = old_mask.unbind()[1].clone()
        else:
            a_mask = torch.ones_like(audio)
        new_audio = audio
        pa = saved.get("audio")
        held_audio = 0
        song_on = bool(cfg.get("song")) or any(cfg.get("placed") or [])
        if cont.get("audio") and pa is not None and not song_on \
                and pa.dim() == audio.dim() and pa.shape[1:-1] == audio.shape[1:-1]:
            ka = min(int(round(n / float(FPS) * AUDIO_LATENT_FPS)),
                     int(pa.shape[-1]), int(audio.shape[-1]) - 1)
            if ka > 0:
                new_audio = audio.clone()
                new_audio[..., :ka] = pa[..., -ka:].to(device=audio.device, dtype=audio.dtype)
                a_mask[..., :ka] = 0.0
                held_audio = ka

        out = dict(latent)
        out["samples"] = comfy.nested_tensor.NestedTensor((new_video, new_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((v_mask, a_mask))

        # guide_frames is retired: it was a check while latent hold was new.
        # The socket stays (outputs connect by position, removing it would
        # shift every wire after it in saved workflows) but nothing is decoded
        # for it any more.
        preview = None

        print("[H3 Studio] latent hold: %d latent frames (%d pictures) from %s%s, "
              "overlap %d"
              % (k, n, os.path.basename(side),
                 ", sound held %.2fs" % (held_audio / float(AUDIO_LATENT_FPS))
                 if held_audio else "", n))
        return out, n, preview

    def _continuation(self, cfg, vae, audio_vae, latent, width, height, frame_count):
        """Anchor the previous clip's tail at frame 0 of this one.

        Writes the same conditioning MiniMaxH3AddGuide writes - one keyframe
        holding the whole run as a single encoded block at resolved_frame_index
        0 - rather than calling that node, so the tail can be taken from the
        right end and levelled before the encode.

        Returns (keyframe or None, frames anchored, the frames themselves).
        The frame count is the
        head that comes back in the output and has to come off before the clip
        is joined to its predecessor.
        """
        cont = cfg.get("cont")
        if not cont:
            return None, 0, None

        want = snap_guide_run(cont["frames"])
        if want < MIN_REF_FRAMES:
            raise ValueError(
                "H3 Studio: a continuation needs at least %d frames. Usable runs "
                "are 5, 22, 39, 56, 73." % MIN_REF_FRAMES)
        if want >= frame_count:
            raise ValueError(
                "H3 Studio: continuing from %d frames into a %d frame clip leaves "
                "nothing new. Shorten the window or lengthen the clip."
                % (want, frame_count))

        # a little slack, then snap down and keep the TAIL - a window that lands
        # between source frames can come back one short or one long, and either
        # way the run must finish on the previous clip's last frame
        # by position, so the run really does finish on the clip's last frame -
        # falls back to the timestamp path for anything not at FPS
        frames = _load_video_tail(cont["file"], want + 17, cont["start"])
        if frames is None:
            frames = _load_video_frames(cont["file"], cont["start"], cont["end"], want + 17)
        # Snap LAST. It used to be min(snap(available), want), which returns
        # `want` untouched whenever want is the smaller of the two - and want
        # comes from the browser's window, which on mkv can round to 23 or 24
        # where the file really holds 22. Anything that is not 17k+5 makes the
        # model reserve rows for one more latent frame than the encode produces,
        # which is the constant shape mismatch: same size every time for a given
        # canvas, a different size for every canvas.
        n = snap_guide_run(min(frames.shape[0], want))
        if n < MIN_REF_FRAMES:
            raise ValueError(
                "H3 Studio: %s gave only %d frames in that window; H3 needs %d. "
                "Widen the window." % (cont["file"], frames.shape[0], MIN_REF_FRAMES))
        frames = frames[-n:]

        frames = flatten_exposure(frames, cont["flatten"])
        frames = _resize(frames, width, height, "center")
        keyframe = {"resolved_frame_index": 0, "latent": vae.encode(frames)}

        # The shape mismatch reports live here. The model reserves rows from the
        # SHAPE of this guide latent and then fills them from the same tensor, so
        # if those two disagree the numbers below say by how much - and whether
        # the guide is a legal 17k+5 run in the first place.
        try:
            print("[H3 Studio] continue: window=%d frames  canvas=%dx%d  "
                  "guide_latent=%s  target_latent=%s  clip_frames=%d"
                  % (n, width, height, tuple(keyframe["latent"].shape),
                     tuple(latent["samples"].tensors[0].shape), frame_count))
        except Exception as e:
            print("[H3 Studio] continue: shape report unavailable (%s)" % e)

        if cont["audio"]:
            # end=None: to EOF, so the carried sound finishes on the clip's last
            # sample rather than at a duration the browser rounded
            # half a second of slack before the window start: the picture run is
            # taken BY POSITION (last n frames) while the trim start is a browser
            # timestamp, so start..EOF can be a frame or two short of the run and
            # the carried sound would stop before the seam does
            track = _load_video_soundtrack(
                cont["file"], max(0.0, cont["start"] - 0.5), None)
            if track is None:
                print("[H3 Studio] continue: %s has no soundtrack, picture only"
                      % cont["file"])
            else:
                # WINDOW THE SOUND THE WAY THE PICTURE IS WINDOWED. The guide is
                # the LAST n frames of the file; this used to hand the model
                # everything from `start` to EOF, which on an untouched trim is the
                # whole clip - up to the entire target audio grid as conditioning,
                # then cropped from the FRONT by max_rt. So the carried sound was
                # the head of the clip while the picture was its tail.
                sr = int(track["sample_rate"])
                wf = track["waveform"]
                keep = max(1, int(round(n / FPS * sr)))
                if wf.shape[-1] > keep:
                    wf = wf[..., -keep:].contiguous()
                track = {"waveform": wf, "sample_rate": sr}

                audio_latent, audio_rt = self._encode_ref_audio(audio_vae, track)
                # anchored at frame 0, so the whole of this clip's audio grid is
                # available; a window this short can only overrun it if the file
                # misreported its rate, and then the TAIL is the half to keep
                max_rt = int(latent["samples"].tensors[1].shape[-1])
                if audio_rt > max_rt:
                    audio_latent = audio_latent[..., -max_rt:].clone()
                keyframe["audio_latent"] = audio_latent
                print("[H3 Studio] continue: guide audio %.3fs -> %d latent frames "
                      "(grid holds %d)"
                      % (wf.shape[-1] / float(sr),
                         int(audio_latent.shape[-1]), max_rt))

        print("[H3 Studio] continue: %d frames from %s at frame 0%s, overlap %d"
              % (n, cont["file"],
                 " (levelled)" if cont["flatten"] > 0 else "", n))
        return keyframe, n, frames

    # ---------------- main ----------------

    def build(self, clip, vae, h3_data, audio_vae=None, first_frame=None, last_frame=None):
        cfg = parse_h3_data(h3_data)

        def finish(cond, refine_cond=None):
            """Shared tail for both modes: attach the continuation, if any.

            refine_cond, when given, is a SECOND encode of the same prompt whose
            image references were built larger (see _refs). It goes out on
            positive_refine instead of the stripped copy of cond."""
            if up:
                refine = refine_cond if refine_cond is not None else cond
                print("[H3 Studio] upscale pass: %s, %d frames%s"
                      % (clip_label(cfg) or "clip", frame_count,
                         (", joins onto %s (%d overlap)"
                          % (os.path.basename(up_source), up_overlap)) if up_source else ""))
                return (latent, up_overlap, up_source, clip_label(cfg), locked_audio,
                        cond, self._without_keyframes(refine),
                        width, height, frame_count, seconds, refine_mul)
            lat = latent
            held = None
            if cfg.get("cont_latent") and cfg.get("cont"):
                held = self._latent_hold(cfg, vae, latent, frame_count)
            if held is not None:
                lat, overlap, guide_frames = held
                guide = None
            else:
                guide, overlap, guide_frames = self._continuation(
                    cfg, vae, audio_vae, latent, width, height, frame_count)
            # exactly what was encoded, so the anchored run can be LOOKED AT
            # instead of inferred from the result of a six minute sample
            if guide_frames is None:
                guide_frames = torch.zeros((1, 64, 64, 3))
            if guide is not None:
                keyframes = list(cond[0][1].get("minimax_keyframes", []))
                keyframes.append(guide)
                cond = node_helpers.conditioning_set_values(
                    cond, {"minimax_keyframes": keyframes})
            self._audit_audio(cond, lat)
            source = ""
            if cfg.get("cont"):
                source = _resolve_asset(cfg["cont"]["file"]) or cfg["cont"]["file"]
            return (lat, overlap, source, clip_label(cfg), locked_audio, cond,
                    self._without_keyframes(refine_cond
                                            if refine_cond is not None else cond),
                    width, height, frame_count, seconds, refine_mul)

        width = max(CANVAS_MULTIPLE, (cfg["width"] // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        height = max(CANVAS_MULTIPLE, (cfg["height"] // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        latent, frame_count = _empty_av_latent(width, height, cfg["length"])
        seconds = frame_count / FPS
        refine_mul = float(cfg.get("ref_refine_scale") or 1.0)
        latent, locked_audio = self._lock_audio(cfg, audio_vae, latent, seconds)

        up = cfg.get("upscale")
        up_source, up_overlap = "", 0
        if up:
            latent, frame_count, width, height = self._load_full(up, cfg)
            seconds = frame_count / FPS
            cfg["cont"] = None                     # the take already holds its join
            up_source = _render_path(up["prev"]) or ""
            up_overlap = up["overlap"] if up_source else 0
            if up["prev"] and not up_source:
                print("[H3 Studio] upscale: the previous clip's upscaled render %r "
                      "is gone - this clip goes out unjoined" % up["prev"]["name"])

        if cfg["mode"] == "fl2va":
            first = first_frame if first_frame is not None else (
                _load_image(cfg["first"]["file"]) if cfg["first"] else None)
            last = last_frame if last_frame is not None else (
                _load_image(cfg["last"]["file"]) if cfg["last"] else None)

            tags, _ = build_tag_map(cfg, first is not None, last is not None)
            prompt = transcribe_prompt(strip_placed_tokens(cfg["prompt"], cfg),
                                       tags, ["@first", "@last"])

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
            return finish(cond)

        # ---- ref2va ----
        known = ([f"@image{i}" for i in range(1, MAX_IMAGES + 1)]
                 + [f"@video{i}" for i in range(1, MAX_VIDEOS + 1)]
                 + [f"@videoaudio{i}" for i in range(1, MAX_VIDEOS + 1)]
                 + [f"@audio{i}" for i in range(1, MAX_AUDIOS + 1)])
        tags, presentation = build_tag_map(cfg, False, False)
        prompt = transcribe_prompt(strip_placed_tokens(inject_carry_note(cfg, tags), cfg),
                                   tags, known)

        ref_items, ref_blocks = self._refs(cfg, vae, audio_vae, width, height, frame_count)
        if presentation:
            print("[H3 Studio] presentation: " + "  ".join("%s %s" % p for p in presentation))

        def _encode(items, blocks):
            tokens = clip.tokenize(prompt, minimax_ref_items=items)
            c = clip.encode_from_tokens_scheduled(tokens)
            if blocks:
                c = node_helpers.conditioning_set_values(c, {"minimax_refs": blocks})
            return c

        cond = _encode(ref_items, ref_blocks)

        # The whole prompt is encoded a second time rather than swapping the
        # latents under the first one: the references go into tokenize() as well
        # as into minimax_refs, so a conditioning holding blocks of one size and
        # tokens built from another is a mismatch waiting to surface inside the
        # sampler. Costs one extra text encode, and only when asked for.
        refine_cond = None
        if refine_mul > 1.0 and any(s is not None for s in cfg["images"]):
            print("[H3 Studio] refine references at %.2gx" % refine_mul)
            r_items, r_blocks = self._refs(cfg, vae, audio_vae, width, height,
                                           frame_count, scale_mul=refine_mul)
            refine_cond = _encode(r_items, r_blocks)

        return finish(cond, refine_cond)

    @classmethod
    def IS_CHANGED(cls, h3_data, **kwargs):
        return h3_data


NODE_CLASS_MAPPINGS = {"CSGlideCastCS": CSGlideCast}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideCastCS": "H3 Studio"}
