"""Glide Join — assemble a continuation onto the clip it continued from.

The anchored frames come back at the HEAD of the new clip, so the same span of
time exists twice: once in the source, once re-rendered by the model. This node
drops one copy and hands the result to Glide Video as a single stream.

Which copy it drops is the whole question, and H3 answers it differently from
LTX. On the LTX chunk render the guide pinned the next chunk's frame 0
PIXEL-IDENTICAL to the previous chunk's overlapped frame, so cutting at the
start of the overlap cost nothing. H3 pins nothing: the anchored frames are
conditioning, and the model re-renders that span in its own colour.

  early_cut  keeps the continuation's version and corrects its colour against
             the source frame by frame. The overlap is then one continuous
             model trajectory, so motion carries better - verified.
  hard_cut   keeps the source's own frames and drops the re-rendered head.
             Nothing to correct, but the join lands at the END of the overlap
             where the two renders have diverged most.

Colour correction is affine and per-frame, because for the length of the
overlap both clips render the SAME INSTANTS and that is ground truth. A single
mean offset assumes the two differ by a constant lift, which is not how
exposure works; fitting gain and offset against the matching frame catches
level, contrast and colour together. Measured against a synthetic
gain-plus-lift flicker: mean matching left 0.021 pixel error, affine left
0.000.
"""

import os

import numpy as np
import torch

try:
    import av
except Exception:  # pragma: no cover
    av = None


SEAM_MODES = ["early_cut", "early_scurve", "hard_cut"]
FIT_SIZE = 48          # frames are averaged to this before fitting
ALIGN_SEARCH = 2       # frames either way when checking correspondence


# --------------------------------------------------------------------------
# source loading
# --------------------------------------------------------------------------

def _open_source(path):
    if av is None:
        raise RuntimeError("Glide Join: PyAV is required to read the source clip.")
    p = (path or "").strip().strip('"').strip("'")
    if not p or not os.path.isfile(p):
        raise FileNotFoundError(
            "Glide Join: source clip not found: %r. Wire H3 Studio's source_video "
            "output, or give a full path." % path)
    return p


def _load_frames(path):
    frames = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError("Glide Join: no video frames decoded from %s" % path)
    return torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)


def _load_audio(path):
    with av.open(path) as container:
        if not container.streams.audio:
            return None
        stream = container.streams.audio[0]
        stream.thread_type = "AUTO"
        rate = int(stream.rate or 48000)
        chunks = []
        for frame in container.decode(audio=0):
            a = frame.to_ndarray()
            if a.ndim == 1:
                a = a[None, :]
            chunks.append(a.astype(np.float32))
    if not chunks:
        return None
    wf = np.concatenate(chunks, axis=-1)
    if wf.max() > 1.5:
        wf = wf / 32768.0
    if wf.shape[0] == 1:
        wf = np.repeat(wf, 2, axis=0)
    return {"waveform": torch.from_numpy(wf)[None, :2], "sample_rate": rate}


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------

def _small(frame):
    """Frame reduced to FIT_SIZE^2 averages, as [P, 3].

    Averaging down before fitting is what makes the fit robust: a highlight the
    model moved lands in one cell instead of skewing the regression, while the
    tone relationship - the thing being measured - survives untouched.
    """
    x = frame.movedim(-1, 0).unsqueeze(0)
    x = torch.nn.functional.adaptive_avg_pool2d(x, FIT_SIZE)
    return x.view(3, -1).movedim(0, 1)


def _affine_fit(a, b, gain_limit=2.0):
    """Per-channel (gain, offset) carrying b onto a, by least squares.

    Degenerate frames - a flat plate with nothing to regress against, or two
    frames that do not actually correspond - fall back to gain 1 and a plain
    mean offset rather than dividing by nothing.
    """
    A, B = _small(a), _small(b)
    ma, mb = A.mean(0), B.mean(0)
    vb = ((B - mb) ** 2).mean(0)
    cov = ((A - ma) * (B - mb)).mean(0)
    g = torch.where(vb > 1e-8, cov / vb.clamp(min=1e-8), torch.ones_like(vb))
    g = g.clamp(1.0 / gain_limit, gain_limit)
    return g, ma - g * mb


def _fit_error(a, b):
    g, o = _affine_fit(a, b)
    return float((_small(a) - (_small(b) * g + o)).abs().mean())


def _best_shift(src, ext, ov):
    """Which ext frame really is src[-ov + k], within +/- ALIGN_SEARCH.

    The per-frame correction rests on ext[k] being the model's version of
    src[-ov + k]. Off by one and every frame is corrected with its neighbour's
    numbers, which reads as flicker rather than as a shift. Cheap to measure:
    fit each candidate alignment and keep the one leaving the least residual.
    """
    n = min(ov, ext.shape[0])
    if n < 3:
        return 0
    probe = [0, n // 2, n - 1]
    best, best_err = 0, None
    for s in range(-ALIGN_SEARCH, ALIGN_SEARCH + 1):
        errs = [_fit_error(src[-ov + k], ext[k + s])
                for k in probe if 0 <= k + s < ext.shape[0]]
        if not errs:
            continue
        err = sum(errs) / len(errs)
        if best_err is None or err < best_err - 1e-6:
            best, best_err = s, err
    return best


def _tracked_affine(src, ext, ov):
    """Per-frame gain and offset across the overlap, held constant after it.

    ext[k] and src[-ov + k] are the same instant rendered twice, so for the
    length of the overlap there is ground truth to fit against. Past it there
    is no reference left, so the last fit is held - a fixed correction, never
    an extrapolated one.

    Inside the overlap the continuation ends up reproducing the source's own
    exposure, including any breathing in it. That is correct: it IS that
    moment, and the source is what the film shows either side of it.
    """
    n = min(ov, ext.shape[0])
    shift = _best_shift(src, ext, ov)
    gains = torch.ones(ext.shape[0], 3, dtype=ext.dtype)
    offs = torch.zeros(ext.shape[0], 3, dtype=ext.dtype)

    done = []
    for k in range(n):
        j = k + shift
        if 0 <= j < ext.shape[0]:
            # the fit belongs to the frame it was measured ON, not to the
            # source frame's index - backwards applies every correction one
            # frame out, which is the flicker it exists to remove
            gains[j], offs[j] = _affine_fit(src[-ov + k], ext[j])
            done.append(j)

    if not done:
        return None, None, shift
    lo, hi = done[0], done[-1]
    for i in range(0, lo):
        gains[i], offs[i] = gains[lo], offs[lo]
    for i in range(hi + 1, ext.shape[0]):
        gains[i], offs[i] = gains[hi], offs[hi]
    return gains, offs, shift


def _scurve(n):
    """Weights 0..1 for the continuation's side of a ramped join.

    tanh k=4: soft at both ends, crosses 50/50 fast, so it spends about two
    frames in the 30-70% ghost zone instead of four. Ramps a velocity
    difference rather than stepping it.
    """
    if n <= 1:
        return torch.ones(max(1, n))
    t = torch.linspace(0.0, 1.0, n)
    w = torch.tanh(4.0 * (t - 0.5))
    return (w - float(w[0])) / (float(w[-1]) - float(w[0]))


class CSGlideJoin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("STRING", {
                    "default": "",
                    "tooltip": "The clip this one continues from. Wire H3 Studio's "
                               "source_video output."}),
                "images": ("IMAGE", {
                    "tooltip": "The continuation's decoded frames, untrimmed - the "
                               "anchored head must still be on them."}),
                "overlap_frames": ("INT", {
                    "default": 22, "min": 0, "max": 4096,
                    "tooltip": "Frames anchored from the source. Wire H3 Studio's "
                               "trim_frames output so it always matches what was "
                               "actually anchored."}),
                "seam_mode": (SEAM_MODES, {
                    "default": "early_cut",
                    "tooltip": "early_cut: keep the continuation's overlap and colour "
                               "it against the source - one continuous model "
                               "trajectory, so motion carries better. early_scurve: "
                               "the same join ramped over a few frames, for when the "
                               "speed change is visible. hard_cut: keep the source's "
                               "own frames instead and drop the re-rendered head."}),
                "seam_blend_frames": ("INT", {
                    "default": 6, "min": 1, "max": 64,
                    "tooltip": "Ramp length for early_scurve. Ignored otherwise."}),
                "match_levels": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Fit the continuation's gain and offset onto the source, "
                               "frame by frame across the overlap and held after it. "
                               "Frame correspondence is checked first, and reported if "
                               "it was out."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "The continuation's audio, untrimmed. Cut to match the "
                               "picture and appended to the source's own track."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                    "tooltip": "Must match what you feed Glide Video."}),
                "match_tail": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Truncate the audio to exactly frames/fps. H3 rounds its "
                               "audio grid up, so every clip ships about 8ms more sound "
                               "than picture and that grows at every join."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT")
    RETURN_NAMES = ("images", "audio", "frame_count")
    FUNCTION = "join"
    CATEGORY = "CGlide"
    DESCRIPTION = ("Join a continuation onto the clip it continued from, matching the "
                   "re-rendered overlap back onto the original's colour.")

    # ---------------- picture ----------------

    def _join_images(self, src, ext, ov, mode, blend, match):
        if ov <= 0:
            return torch.cat([src, ext], dim=0)
        if ov > src.shape[0]:
            raise ValueError("Glide Join: overlap is %d frames but the source only "
                             "has %d." % (ov, src.shape[0]))
        if ov >= ext.shape[0]:
            raise ValueError("Glide Join: overlap is %d frames but the continuation "
                             "only has %d. Are these frames already trimmed?"
                             % (ov, ext.shape[0]))
        if src.shape[1:] != ext.shape[1:]:
            raise ValueError("Glide Join: source is %dx%d and the continuation is "
                             "%dx%d. Render both at the same canvas."
                             % (src.shape[2], src.shape[1], ext.shape[2], ext.shape[1]))

        if match and mode == "hard_cut":
            # the source's own frames survive, so one fit measured at the cut is
            # all the continuation needs
            g, o = _affine_fit(src[-1], ext[ov - 1])
            ext = (ext * g + o).clamp(0.0, 1.0)
            print("[Glide Join] level matched at the cut, gain %.3f %.3f %.3f, "
                  "offset %.4f %.4f %.4f"
                  % (float(g[0]), float(g[1]), float(g[2]),
                     float(o[0]), float(o[1]), float(o[2])))
        elif match:
            gains, offs, shift = _tracked_affine(src, ext, ov)
            if shift:
                print("[Glide Join] frame correspondence was off by %+d - corrected "
                      "here, but the anchor window is a frame out." % shift)
            if gains is not None:
                ext = (ext * gains.view(-1, 1, 1, 3)
                       + offs.view(-1, 1, 1, 3)).clamp(0.0, 1.0)
                n = min(ov, ext.shape[0])
                print("[Glide Join] level tracked over %d frames, gain %.3f..%.3f, "
                      "held at %.3f %.3f %.3f / %.4f %.4f %.4f"
                      % (n, float(gains[:n].min()), float(gains[:n].max()),
                         float(gains[-1][0]), float(gains[-1][1]), float(gains[-1][2]),
                         float(offs[-1][0]), float(offs[-1][1]), float(offs[-1][2])))

        if mode == "hard_cut":
            return torch.cat([src, ext[ov:]], dim=0)

        head = src[:-ov]
        if mode == "early_cut":
            return torch.cat([head, ext], dim=0)

        n = min(max(1, blend), ov, ext.shape[0])
        w = _scurve(n).view(-1, 1, 1, 1).to(ext.dtype)
        mixed = src[-ov:-ov + n] * (1.0 - w) + ext[:n] * w
        return torch.cat([head, mixed, ext[n:]], dim=0)

    # ---------------- sound ----------------

    def _join_audio(self, src_audio, ext_audio, cut_frames, ext_skip_frames,
                    total_frames, fps, match_tail):
        if src_audio is None and ext_audio is None:
            return None

        rate = int((src_audio or ext_audio)["sample_rate"])

        def wave(a):
            if a is None:
                return None
            if int(a["sample_rate"]) != rate:
                raise ValueError("Glide Join: the source is %d Hz and the "
                                 "continuation is %d Hz. Resample one first."
                                 % (int(a["sample_rate"]), rate))
            wf = a["waveform"]
            return wf[0] if wf.dim() == 3 else wf

        head, tail = wave(src_audio), wave(ext_audio)
        parts = []

        if head is not None:
            cut = int(round(cut_frames / fps * rate))
            if cut > head.shape[-1]:
                head = torch.cat(
                    [head, torch.zeros(head.shape[0], cut - head.shape[-1],
                                       dtype=head.dtype)], dim=-1)
            parts.append(head[..., :cut])

        if tail is not None:
            # whatever picture came off the continuation's head comes off its
            # sound too, or the track runs a full overlap long and match_tail
            # then cuts that from the END - doubling sound at the join and
            # losing the same span from the finish
            skip = int(round(max(0, ext_skip_frames) / fps * rate))
            if skip < tail.shape[-1]:
                parts.append(tail[..., skip:] if skip else tail)

        if not parts:
            return None
        if len(parts) == 2 and parts[0].shape[0] != parts[1].shape[0]:
            ch = min(parts[0].shape[0], parts[1].shape[0])
            parts = [p[:ch] for p in parts]
        wf = torch.cat(parts, dim=-1)

        if match_tail:
            want = int(round(total_frames / fps * rate))
            have = int(wf.shape[-1])
            if have > want:
                print("[Glide Join] tail trimmed %d samples (%.2f ms)"
                      % (have - want, (have - want) / rate * 1000.0))
                wf = wf[..., :want]
            elif have < want:
                wf = torch.cat(
                    [wf, torch.zeros(wf.shape[0], want - have, dtype=wf.dtype)],
                    dim=-1)

        return {"waveform": wf[None], "sample_rate": rate}

    # ---------------- main ----------------

    def join(self, source_video, images, overlap_frames, seam_mode,
             seam_blend_frames, match_levels, audio=None, fps=24.0, match_tail=True):
        # No source clip means this is not a continuation at all - the CONTINUE
        # FROM slot is empty, so H3 Studio sends "" and 0 - and the graph is
        # making an ordinary clip. Pass the frames straight through instead of
        # failing, so ONE workflow does both jobs and nothing has to be rewired
        # or bypassed between a normal generation and a chained one.
        if not (source_video or "").strip():
            print("[Glide Join] no source clip - passing %d frames through "
                  "unchanged (not a continuation)" % images.shape[0])
            return (images, audio, int(images.shape[0]))

        path = _open_source(source_video)
        src = _load_frames(path)
        ov = max(0, int(overlap_frames))

        out = self._join_images(src, images, ov, seam_mode,
                                int(seam_blend_frames), bool(match_levels))

        if seam_mode == "hard_cut":
            cut_frames, ext_skip = src.shape[0], ov
        else:
            cut_frames, ext_skip = src.shape[0] - ov, 0
        out_audio = self._join_audio(_load_audio(path), audio, cut_frames, ext_skip,
                                     out.shape[0], float(fps), bool(match_tail))

        print("[Glide Join] %s: %d + %d frames, overlap %d -> %d frames (%.3fs)"
              % (seam_mode, src.shape[0], images.shape[0], ov,
                 out.shape[0], out.shape[0] / float(fps)))
        return (out, out_audio, int(out.shape[0]))

    @classmethod
    def IS_CHANGED(cls, source_video, **kwargs):
        try:
            p = _open_source(source_video)
            return "%s:%d" % (p, os.stat(p).st_mtime_ns)
        except Exception:
            return ""


NODE_CLASS_MAPPINGS = {"CSGlideJoinCS": CSGlideJoin}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideJoinCS": "Glide Join"}
