"""
csglide_assemble.py -- Assemble film

Puts a project's clips together into one file, once, at the end.

Why this exists: with Glide Join on output = film, every link decoded the
whole film so far, held it as float frames and encoded all of it again. At
1080p that is ~25 MB a frame, so a one-minute film needed ~35 GB of RAM for a
single join and the cost grew with every clip. With output = clip each clip's
file holds only that clip, and the film is made here instead:

    clip 1 [minus what clip 2 replaced] + clip 2 [minus what clip 3 replaced] + ...

How much of each clip to keep is read off the NEXT clip's own file: Glide Join
writes a "[cglide-join] ... trim_prev=N" line into the chain note, and Glide
Video stores that note in the file. A file that says output=film already holds
everything before it, so the film restarts from it.

Frames are streamed: one clip decoded at a time, one frame at a time, straight
into the same ffmpeg command Glide Video builds. Memory stays flat whatever the
length. The sound is cut on the same frame boundaries, sample exact, and goes
in as one PCM track.

Same pixels as output = film: the 16-bit frames decoded here are the ones
Glide Join would have decoded, and Glide Video's float -> 16-bit step gives the
same integers back.
"""

import collections
import os
import re
import tempfile
import threading

import numpy as np

try:
    import av
except Exception:  # pragma: no cover
    av = None

try:
    import folder_paths
except ImportError:  # outside ComfyUI, for tests
    folder_paths = None

try:
    from . import csglide_run as _run
    from .csglide_video_presets import (build_ffmpeg_cmd, needs_preview_copy,
                                        pipe_format, probe_video,
                                        resolve_container, resolve_preset,
                                        write_ffmetadata)
    from .csglide_video import _drain, _next_path
    from .csglide_join import JOIN_MARK, _load_audio, _resample
except ImportError:
    import csglide_run as _run
    from csglide_video_presets import (build_ffmpeg_cmd, needs_preview_copy,
                                       pipe_format, probe_video,
                                       resolve_container, resolve_preset,
                                       write_ffmetadata)
    from csglide_video import _drain, _next_path
    from csglide_join import JOIN_MARK, _load_audio, _resample

import torch

_MARK_RE = re.compile(re.escape(JOIN_MARK) + r"\s+(.*)$")
STDERR_LINES = 400


# ---------------------------------------------------------------------
# reading clips
# ---------------------------------------------------------------------

def read_mark(path):
    """The clip's own join line: {"output", "trim_prev", "overlap", "mode"}, or
    None when the file carries none (made before this existed, or by a
    Glide Join that was bypassed / had nothing to join to).

    The LAST mark in the note is this clip's: the note carries the history
    of every clip before it, and Glide Join appends its own line after it."""
    try:
        with av.open(path) as c:
            meta = c.metadata or {}
            note = meta.get("description") or meta.get("DESCRIPTION") or ""
    except Exception:
        return None
    found = None
    for line in str(note).splitlines():
        m = _MARK_RE.search(line.strip())
        if m:
            found = m.group(1)
    if found is None:
        return None
    out = {}
    for part in found.split():
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
    try:
        return {"output": out.get("output", "clip"),
                "trim_prev": int(out.get("trim_prev", "0")),
                "overlap": int(out.get("overlap", "0")),
                "mode": out.get("mode", "")}
    except ValueError:
        return None


def describe(path):
    """(frames, width, height, fps, description) without decoding pictures:
    frames are counted from packets, which is exact for video (one packet,
    one frame) and costs a read of the file, not a decode."""
    with av.open(path) as c:
        st = c.streams.video[0]
        w, h = int(st.codec_context.width), int(st.codec_context.height)
        fps = float(st.average_rate) if st.average_rate else 0.0
        meta = c.metadata or {}
        desc = meta.get("description") or meta.get("DESCRIPTION") or ""
        n = 0
        for pkt in c.demux(st):
            if pkt.size:
                n += 1
    return n, w, h, fps, str(desc)


def _note(path):
    try:
        with av.open(path) as c:
            meta = c.metadata or {}
            return str(meta.get("description") or meta.get("DESCRIPTION") or "")
    except Exception:
        return ""


def plan(clips, log=print):
    """Which file contributes how many frames, in order.

    clips: [{"path", "linked", "frames" (the clip's own length in the
    project), "overlap" (the project's overlap for this link), "label"}]
    Returns [{"path", "keep", "frames", "w", "h", "fps", "label"}].
    """
    segs = []
    for k, c in enumerate(clips):
        n, w, h, fps, desc = describe(c["path"])
        mark = read_mark(c["path"])
        label = c.get("label") or os.path.basename(c["path"])
        whole = False
        trim = 0
        if mark is not None:
            whole = mark["output"] == "film"
            trim = mark["trim_prev"] if c.get("linked") else 0
            if mark["trim_prev"] and not c.get("linked"):
                log("[Assemble] %s replaced %d frames of the clip before it, but it "
                    "is not linked in the project any more - kept both whole"
                    % (label, mark["trim_prev"]))
        else:
            # No mark: a render from before output = clip existed. A file much
            # longer than its clip is a whole film (output = film did that).
            own = int(c.get("frames") or 0)
            if c.get("linked") and own and n > own + 24:
                whole = True
            elif c.get("linked"):
                trim = int(c.get("overlap") or 0)
                log("[Assemble] %s has no join line (rendered before output = clip?) - "
                    "assuming it replaced the last %d frames of the clip before it"
                    % (label, trim))
        if whole:
            if segs:
                log("[Assemble] %s is a whole-film render (Glide Join output = film): "
                    "it already holds the %d clip(s) before it, starting from it"
                    % (label, len(segs)))
            segs = []
            trim = 0
        if trim and segs:
            prev = segs[-1]
            if trim > prev["keep"]:
                raise ValueError("Assemble: %s replaces %d frames but %s only has %d"
                                 % (label, trim, prev["label"], prev["keep"]))
            prev["keep"] -= trim
        segs.append({"path": c["path"], "keep": n, "frames": n, "w": w, "h": h,
                     "fps": fps, "label": label})
    if not segs:
        raise ValueError("Assemble: no clips to put together")
    # The film's size is the one MOST clips share, not the first clip's: one
    # odd render at the start used to make every other clip look wrong.
    sizes = {}
    for sg in segs:
        sizes[(sg["w"], sg["h"])] = sizes.get((sg["w"], sg["h"]), 0) + 1
    w, h = max(sizes, key=lambda k: (sizes[k], -segs.index(next(
        sg for sg in segs if (sg["w"], sg["h"]) == k))))
    bad = [sg for sg in segs if (sg["w"], sg["h"]) != (w, h)]
    if bad:
        raise ValueError(
            "Assemble: %s %s a different size from the rest of the film (%dx%d). "
            "Right-click %s in the strip -> Takes and pick a %dx%d render, or render "
            "%s again. (A video bigger than its latent is usually an old render made "
            "with a refine pass inside the draft workflow.)"
            % (", ".join("%s (%s, %dx%d)" % (sg["label"], os.path.basename(sg["path"]),
                                             sg["w"], sg["h"]) for sg in bad),
               "is" if len(bad) == 1 else "are", w, h,
               "it" if len(bad) == 1 else "each", w, h,
               "it" if len(bad) == 1 else "them"))
    return segs


# ---------------------------------------------------------------------
# sound
# ---------------------------------------------------------------------

def build_audio(segs, fps):
    """One track: each clip's sound cut on the same frame boundary as its
    picture, sample exact. A clip without sound contributes silence of its
    length, so the ones after it stay in sync. Returns (wave [C, N], rate)
    or (None, None) when no clip has sound."""
    loaded = []
    rate = None
    for s in segs:
        try:
            a = _load_audio(s["path"])
        except Exception:
            a = None
        if a is not None and rate is None:
            rate = int(a["sample_rate"])
        loaded.append(a)
    if rate is None:
        return None, None
    parts = []
    for s, a in zip(segs, loaded):
        want = int(round(s["keep"] / float(fps) * rate))
        if a is None:
            parts.append(torch.zeros(2, want))
            continue
        wf = a["waveform"]
        wf = wf[0] if wf.dim() == 3 else wf
        if int(a["sample_rate"]) != rate:
            wf = _resample(wf, int(a["sample_rate"]), rate)
        wf = wf[:2].float()
        if wf.shape[0] == 1:
            wf = wf.repeat(2, 1)
        if wf.shape[-1] >= want:
            wf = wf[..., :want]
        else:
            wf = torch.cat([wf, torch.zeros(wf.shape[0], want - wf.shape[-1])], dim=-1)
        parts.append(wf)
    return torch.cat(parts, dim=-1), rate


# ---------------------------------------------------------------------
# picture
# ---------------------------------------------------------------------

def frames_of(segs, deep):
    """Every kept frame, in order, as the raw bytes ffmpeg's pipe wants -
    one frame in memory at a time.

    Always decoded at 16 bits, like Glide Join does. For an 8-bit preset the
    step down is Glide Video's own (x / 65535 * 255, rounded), so an 8-bit
    film comes out with the same integers output = film would have given."""
    fmt = "rgb48le"
    for s in segs:
        left = s["keep"]
        if left <= 0:
            continue
        with av.open(s["path"]) as c:
            st = c.streams.video[0]
            st.thread_type = "AUTO"
            for frame in c.decode(video=0):
                if left <= 0:
                    break
                a = frame.to_ndarray(format=fmt)
                if deep:
                    a = a.astype("<u2", copy=False)
                else:
                    f = a.astype(np.float32)
                    f /= 65535.0
                    a = np.clip(f * 255.0 + 0.5, 0, 255).astype(np.uint8)
                yield np.ascontiguousarray(a).tobytes()
                left -= 1
        if left > 0:
            raise ValueError("Assemble: %s decoded %d frames short of the %d counted"
                             % (s["label"], left, s["keep"]))


def _encode(cmd, chunks):
    proc = _run.popen(cmd[0], cmd[1:])
    lines = collections.deque(maxlen=STDERR_LINES)
    pump = threading.Thread(target=_drain, args=(proc.stderr, lines), daemon=True)
    pump.start()
    written = 0
    try:
        for b in chunks:
            try:
                proc.stdin.write(b)
            except (BrokenPipeError, OSError):
                break
            written += 1
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
    except Exception:
        proc.kill()
        pump.join(timeout=2.0)
        raise
    code = proc.wait()
    pump.join(timeout=5.0)
    return code, "\n".join(lines), written


# ---------------------------------------------------------------------
# the whole thing
# ---------------------------------------------------------------------

def assemble(clips, preset, container="auto", fps=24.0, prefix="film/film",
             out_dir=None, log=print):
    """Plan, cut the sound, stream the pictures through one encode.
    Returns {"path", "name", "subfolder", "frames", "seconds", "segments"}."""
    if av is None:
        raise RuntimeError("Assemble: PyAV is required.")
    segs = plan(clips, log)
    fps = float(fps or segs[0]["fps"] or 24.0)
    total = sum(s["keep"] for s in segs)
    for s in segs:
        log("[Assemble] %-24s keeps %4d of %4d frames"
            % (s["label"], s["keep"], s["frames"]))

    out_dir = out_dir or (folder_paths.get_output_directory() if folder_paths
                          else os.path.abspath("output"))
    ext, _ = resolve_container(preset, container)
    out_path, filename, subdir = _next_path(out_dir, prefix, ext)

    # Same browser-playable copy Glide Video makes for a master the browser
    # cannot decode, so the film can be shown in Glide Video's own player.
    preview_path = preview_name = None
    if needs_preview_copy(preset):
        tdir = (folder_paths.get_temp_directory() if folder_paths
                else tempfile.gettempdir())
        os.makedirs(tdir, exist_ok=True)
        preview_name = os.path.splitext(filename)[0] + "_preview.mp4"
        preview_path = os.path.join(tdir, preview_name)

    tmp = tempfile.mkdtemp(prefix="cglide_assemble_")
    pcm = meta = None
    try:
        wave, rate = build_audio(segs, fps)
        channels = 2
        if wave is not None:
            pcm = os.path.join(tmp, "film.pcm")
            channels = int(wave.shape[0])
            with open(pcm, "wb") as f:
                f.write(np.ascontiguousarray(wave.numpy().T.astype(np.float32)).tobytes())
        notes = [d for d in [_note(segs[-1]["path"])] if d]
        meta = os.path.join(tmp, "film.txt")
        write_ffmetadata(meta, {
            "encoder": "Glide Video (Assemble film)",
            "description": "\n".join(notes + [
                "%s assembled: %s" % (JOIN_MARK, " + ".join(
                    "%s[:%d]" % (os.path.basename(s["path"]), s["keep"]) for s in segs))]),
        })
        cmd = build_ffmpeg_cmd(preset=preset, width=segs[0]["w"], height=segs[0]["h"],
                               fps=fps, out_path=out_path, audio_path=pcm,
                               audio_rate=rate, audio_channels=channels,
                               preview_path=preview_path, meta_path=meta, container=ext)
        deep = pipe_format(preset) == "rgb48le"
        code, err, written = _encode(cmd, frames_of(segs, deep))
        if code != 0:
            raise RuntimeError("Assemble: ffmpeg failed (exit %s) after %d of %d frames\n%s"
                               % (code, written, total, err.strip()))
    finally:
        for p in (pcm, meta):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        try:
            os.rmdir(tmp)
        except OSError:
            pass

    info = probe_video(out_path)
    log("[Assemble] %s: %d clip(s), %d frames (%.2fs)  ->  %s"
        % (filename, len(segs), total, total / fps,
           " / ".join(v for v in (info.get("codec"), info.get("pix_fmt")) if v)))
    sub = (subdir or "").replace("\\", "/")
    stream = " / ".join(v for v in (info.get("codec"), info.get("profile"),
                                    info.get("pix_fmt")) if v)
    return {"path": out_path, "name": filename, "subfolder": sub,
            "frames": total, "seconds": total / fps,
            "segments": [{"file": os.path.basename(s["path"]), "keep": s["keep"],
                          "of": s["frames"]} for s in segs],
            # exactly the entry Glide Video sends after a render, so its
            # player can show the film like any other output
            "ui": {"filename": preview_name or filename,
                   "subfolder": "" if preview_name else sub,
                   "type": "temp" if preview_name else "output",
                   "fps": fps, "frames": total,
                   "width": segs[0]["w"], "height": segs[0]["h"],
                   "format": "video/mp4" if (preview_name or filename).endswith(".mp4")
                             else "video/" + ext,
                   "master": filename, "master_subfolder": sub,
                   "master_type": "output", "stream": stream}}
