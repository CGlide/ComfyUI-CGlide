"""H3 Studio — one node covering both MiniMax H3 model families.

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
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
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
        }

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
        "cont": cont,
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
            print("[H3 Studio] %s points at an empty slot — removed from the prompt." % token)
            out = out.replace(token, "")
    return out


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

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

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT",
                    "STRING", "IMAGE")
    RETURN_NAMES = ("positive", "latent", "width", "height", "length", "seconds",
                    "overlap_frames", "source_video", "guide_frames")
    FUNCTION = "build"
    CATEGORY = "CGlide"
    DESCRIPTION = "MiniMax H3 director — first/last keyframes or omni references, with automatic reference tagging."

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
            # end=None: to EOF, so the carried sound finishes where the
            # picture does rather than at a duration the browser rounded
            track = _load_video_soundtrack(cont["file"], cont["start"], None)
            if track is None:
                print("[H3 Studio] continue: %s has no soundtrack, picture only"
                      % cont["file"])
            else:
                audio_latent, audio_rt = self._encode_ref_audio(audio_vae, track)
                # anchored at frame 0, so the whole of this clip's audio grid is
                # available; crop only if the window somehow overruns it
                max_rt = int(latent["samples"].tensors[1].shape[-1])
                if audio_rt > max_rt:
                    audio_latent = audio_latent[..., :max_rt].clone()
                keyframe["audio_latent"] = audio_latent

        print("[H3 Studio] continue: %d frames from %s at frame 0%s, overlap %d"
              % (n, cont["file"],
                 " (levelled)" if cont["flatten"] > 0 else "", n))
        return keyframe, n, frames

    # ---------------- main ----------------

    def build(self, clip, vae, h3_data, audio_vae=None, first_frame=None, last_frame=None):
        cfg = parse_h3_data(h3_data)

        def finish(cond):
            """Shared tail for both modes: attach the continuation, if any."""
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
            source = ""
            if cfg.get("cont"):
                source = _resolve_asset(cfg["cont"]["file"]) or cfg["cont"]["file"]
            return (cond, latent, width, height, frame_count, seconds, overlap,
                    source, guide_frames)

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
            return finish(cond)

        # ---- ref2va ----
        known = ([f"@image{i}" for i in range(1, MAX_IMAGES + 1)]
                 + [f"@video{i}" for i in range(1, MAX_VIDEOS + 1)]
                 + [f"@videoaudio{i}" for i in range(1, MAX_VIDEOS + 1)]
                 + [f"@audio{i}" for i in range(1, MAX_AUDIOS + 1)])
        tags, presentation = build_tag_map(cfg, False, False)
        prompt = transcribe_prompt(cfg["prompt"], tags, known)

        ref_items, ref_blocks = self._refs(cfg, vae, audio_vae, width, height, frame_count)
        if presentation:
            print("[H3 Studio] presentation: " + "  ".join("%s %s" % p for p in presentation))

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return finish(cond)

    @classmethod
    def IS_CHANGED(cls, h3_data, **kwargs):
        return h3_data


NODE_CLASS_MAPPINGS = {"CSGlideCastCS": CSGlideCast}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideCastCS": "H3 Studio"}
