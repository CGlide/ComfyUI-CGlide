"""
csglide_video_presets.py

Encoder presets for the CGlide video combine node.

Three share presets, one master, one archive. Each is a single balanced
choice -- no crf/preset/pix_fmt widgets exposed to the user. Quality
targets are matched across codecs so switching preset changes file size,
not how it looks.

Usage:

    from cglide_video_presets import (
        available_presets, build_ffmpeg_cmd, PRESETS
    )

    names = available_presets()          # probes ffmpeg, caches result
    cmd = build_ffmpeg_cmd(
        preset="AV1 (small, best quality)",
        width=1344, height=768, fps=24.0,
        audio_path="/tmp/run.wav",        # or None
        out_path="/output/clip.mp4",
    )
    # frames go to proc.stdin as raw rgb24 bytes

Frames are piped in as raw rgb24 on stdin and audio comes in as a second
input, so there is exactly one ffmpeg process and exactly one output
file. That single-pass shape is what avoids the silent-duplicate problem
in encode-then-remux designs.
"""

import shutil
import subprocess

# ---------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------
#
# Quality notes, for whoever maintains this later:
#
#   The share presets use CAPPED CRF: a quality target with a bitrate
#   ceiling. CRF decides how good it looks and spends only what the
#   content needs -- a locked-off shot costs less than a whip pan, which
#   is the whole point. -maxrate / -bufsize then stop a genuinely hard
#   sequence from spiking past what a player can stream.
#
#   Consequence worth knowing: a clean, simple clip will report a low
#   bitrate. That is CRF working, not the encoder short-changing you. A
#   fixed-rate encoder would have spent the same bits on an easy shot as
#   on a hard one and looked no better.
#
#   CRF/CQ scales are NOT comparable between codecs. The numbers below
#   are each tuned to land near "visually indistinguishable from source"
#   for diffusion video with headroom to spare, which is why they differ
#   so much. Roughly, every 4 points of CRF halves or doubles the rate:
#   drop 4 for a near-transparent handoff master, raise 4 for smaller
#   share files.
#
#   Ceilings are set for the native 1344x768 canvas. They only bind on
#   hard content, so at draft sizes they simply never engage.
#
#   10-bit (yuv420p10le) on the AV1 preset is not a typo. Encoding 8-bit
#   source at 10-bit depth gives smaller files at equal quality, because
#   the encoder's internal precision improves and it stops spending bits
#   correcting banding. Gradients -- skies, sand, fog -- benefit most.
#
#   yuv420p is correct for share presets: hardware decoders in phones
#   and TVs frequently support nothing else. Masters use 4:2:2 / 4:4:4
#   because chroma subsampling damage concentrates on saturated colour
#   edges, which is exactly what diffusion footage is full of.

PRESETS = {
    # -------- share --------
    "H.264 (compatible)": {
        "encoder": "libx264",
        "container": "mp4",
        "browser_playable": True,
        "note": "Plays everywhere. Largest of the three share presets.",
        "args": [
            "-c:v", "libx264",
            "-crf", "15",
            "-maxrate", "16000k",
            "-bufsize", "24000k",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
        ],
    },

    "H.265 (smaller)": {
        "encoder": "libx265",
        "container": "mp4",
        "browser_playable": True,
        "note": "~35% smaller than H.264 at the same look. Grain-preserving tuning applied.",
        "args": [
            "-c:v", "libx265",
            "-crf", "19",
            "-maxrate", "13000k",
            "-bufsize", "20000k",
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            # HEVC's default deblocking and SAO smear fine detail --
            # exactly the grain and texture diffusion output is prized
            # for. Untuned x265 can look WORSE than H.264 at equal size.
            "-x265-params", "deblock=-2,-2:no-sao=1:selective-sao=0",
            # Some players only accept the hvc1 tag; x265 defaults to
            # hev1. This one flag prevents most "saves but won't play"
            # reports.
            "-tag:v", "hvc1",
        ],
    },

    "AV1 (small, best quality)": {
        "encoder": "av1_nvenc",
        "container": "mp4",
        "browser_playable": True,
        "note": "GPU encoded, near-instant. Smallest share preset, best gradients.",
        "fallback": "AV1 (small, CPU)",
        "args": [
            "-c:v", "av1_nvenc",
            "-preset", "p6",
            "-tune", "hq",
            "-rc", "vbr",
            "-cq", "24",
            # -b:v 0 is REQUIRED alongside -cq. Without it NVENC silently
            # ignores the quality target and falls back to a default
            # bitrate. The ceiling below still applies.
            "-b:v", "0",
            "-maxrate", "10000k",
            "-bufsize", "16000k",
            "-pix_fmt", "yuv420p10le",
        ],
    },

    "AV1 (small, CPU)": {
        "encoder": "libsvtav1",
        "container": "mp4",
        "browser_playable": True,
        "hidden": True,  # automatic fallback when NVENC AV1 is absent
        "note": "CPU AV1. Better quality per byte than NVENC, much slower.",
        "args": [
            "-c:v", "libsvtav1",
            "-crf", "20",
            "-maxrate", "10000k",
            "-bufsize", "16000k",
            "-preset", "5",
            "-pix_fmt", "yuv420p10le",
            # tune=0 optimises for how it looks; the default optimises
            # for PSNR, which is not the same thing.
            "-svtav1-params", "tune=0",
        ],
    },

    # -------- high fidelity --------
    #
    # 4:2:0 discards three quarters of the colour resolution BEFORE the
    # encoder starts. That loss is fixed and no bitrate recovers it, and
    # it lands on saturated colour edges -- which is what diffusion
    # footage is made of. These two keep all of it.
    #
    # AV1 is deliberately absent: libsvtav1 has no 4:4:4 mode, av1_nvenc
    # advertises one that is hardware-gated and unreliable, and libaom is
    # too slow to be worth offering. At 4:4:4 10-bit the codec hardly
    # matters anyway.

    "H.264 4:4:4 10-bit": {
        "encoder": "libx264",
        "container": "mp4",
        "browser_playable": False,
        "pipe": "rgb48le",
        "note": "Full colour, 10-bit. For chunk handoff and grading. Editors yes, phones no.",
        "args": [
            "-c:v", "libx264",
            "-crf", "12",
            "-preset", "medium",
            "-pix_fmt", "yuv444p10le",
            "-profile:v", "high444",
        ],
    },

    "H.265 4:4:4 10-bit": {
        "encoder": "libx265",
        "container": "mkv",
        "browser_playable": False,
        "pipe": "rgb48le",
        "note": "Same fidelity as the H.264 4:4:4 preset, appreciably smaller.",
        "args": [
            "-c:v", "libx265",
            "-crf", "16",
            "-preset", "medium",
            "-pix_fmt", "yuv444p10le",
            "-x265-params", "deblock=-2,-2:no-sao=1:selective-sao=0",
        ],
    },

    # -------- master / archive --------
    "ProRes 422 HQ (master)": {
        "encoder": "prores_ks",
        "container": "mov",
        "browser_playable": False,
        "pipe": "rgb48le",
        "note": "Editing master. All-intra, frame-accurate seeking, no generation loss.",
        "args": [
            "-c:v", "prores_ks",
            "-profile:v", "3",
            "-vendor", "apl0",
            "-pix_fmt", "yuv422p10le",
        ],
    },

    "FFV1 (lossless archive)": {
        "encoder": "ffv1",
        "container": "mkv",
        "browser_playable": False,
        "pipe": "rgb48le",
        "note": "Bit-exact. No colour conversion at all. Large files.",
        "args": [
            "-c:v", "ffv1",
            "-level", "3",
            "-g", "1",
            # gbrp16le, NOT yuv444p16le. The frames arrive as RGB, and any
            # YUV target means an irreversible matrix conversion -- measured
            # at max error 33/65535 on a gradient, which makes "lossless"
            # untrue. Staying in RGB round-trips bit-exact AND is ~36%
            # smaller, because there is no conversion error to encode.
            "-pix_fmt", "gbrp16le",
        ],
    },
}

DEFAULT_PRESET = "H.264 (compatible)"

# Audio is identical across every preset -- there is no reason to vary it.
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ac", "2"]


# ---------------------------------------------------------------------
# Encoder availability
# ---------------------------------------------------------------------

_encoder_cache = None
_ffmpeg_cache = None


def _ffmpeg_candidates():
    """Every place ffmpeg might reasonably be, best first.

    Windows does not ship ffmpeg and ComfyUI portable does not add it to
    PATH, so shutil.which() alone fails on most Windows installs.
    """
    import os

    # 1. explicit override wins
    env = os.environ.get("CSGLIDE_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if env:
        yield env

    # 2. on PATH
    w = shutil.which("ffmpeg")
    if w:
        yield w

    # 3. imageio-ffmpeg ships a working binary and is a common transitive
    #    dependency in ComfyUI installs -- usually present even when the
    #    system has no ffmpeg at all
    try:
        import imageio_ffmpeg
        yield imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    # 4. relative to the ComfyUI tree, and the usual Windows install spots
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.abspath(os.path.join(here, "..", "..")),        # ComfyUI/
        os.path.abspath(os.path.join(here, "..", "..", "..")),  # portable root
    ]
    names = ["ffmpeg.exe", "ffmpeg"]
    subs = ["", "ffmpeg", os.path.join("ffmpeg", "bin"), "bin",
            os.path.join("python_embeded", "Scripts")]
    for root in roots:
        for sub in subs:
            for name in names:
                yield os.path.join(root, sub, name) if sub else os.path.join(root, name)

    for p in [r"C:\ffmpeg\bin\ffmpeg.exe",
              r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        yield p


def ffmpeg_path(refresh=False):
    """Locate a working ffmpeg, caching the result."""
    global _ffmpeg_cache
    if _ffmpeg_cache is not None and not refresh:
        return _ffmpeg_cache

    import os
    for cand in _ffmpeg_candidates():
        if not cand:
            continue
        try:
            if os.path.sep in cand and not os.path.isfile(cand):
                continue
            subprocess.run([cand, "-version"], capture_output=True, timeout=10)
            _ffmpeg_cache = cand
            return cand
        except Exception:
            continue

    _ffmpeg_cache = "ffmpeg"   # last resort; will fail loudly at encode time
    return _ffmpeg_cache


def installed_encoders(refresh=False):
    """Set of encoder names this ffmpeg build actually has.

    ComfyUI's bundled ffmpeg is frequently built without NVENC, so this
    must be probed rather than assumed from the GPU.
    """
    global _encoder_cache
    if _encoder_cache is not None and not refresh:
        return _encoder_cache

    import re

    found = set()
    exe = ffmpeg_path()
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        # encoder lines look like: " V....D av1_nvenc   NVIDIA NVENC av1 encoder"
        for m in re.finditer(r"^\s*([VAS][\.A-Z]{5})\s+(\S+)", out, re.M):
            name = m.group(2)
            if name != "=":
                found.add(name)
    except Exception as e:
        print("[Glide Video] could not run ffmpeg at %r: %s" % (exe, e))

    if not found:
        print("[Glide Video] no ffmpeg encoders detected -- only H.264 will be "
              "offered, and encoding will fail. Install ffmpeg, or set the "
              "CSGLIDE_FFMPEG environment variable to its full path.")
    else:
        wanted = ["libx264", "libx265", "libsvtav1", "av1_nvenc",
                  "hevc_nvenc", "h264_nvenc", "prores_ks", "ffv1"]
        have = [w for w in wanted if w in found]
        print("[Glide Video] ffmpeg: %s (%d encoders; %s)"
              % (exe, len(found), ", ".join(have) or "none of the expected"))

    _encoder_cache = found
    return found


def resolve_preset(name):
    """Return (preset_name, preset_dict), following fallbacks.

    Falls through to the preset's declared fallback if its encoder is
    missing, then to DEFAULT_PRESET.
    """
    have = installed_encoders()
    seen = set()

    while name in PRESETS and name not in seen:
        seen.add(name)
        p = PRESETS[name]
        if p["encoder"] in have:
            return name, p
        name = p.get("fallback")

    return DEFAULT_PRESET, PRESETS[DEFAULT_PRESET]


def available_presets(include_hidden=False):
    """Preset names to show in the node's dropdown.

    A preset is listed if its own encoder exists, or if any preset it
    falls back to exists -- so "AV1 (small, best quality)" still appears
    on a machine without NVENC and quietly uses CPU AV1.
    """
    have = installed_encoders()
    names = []
    for name, p in PRESETS.items():
        if p.get("hidden") and not include_hidden:
            continue
        resolved, _ = resolve_preset(name)
        if PRESETS[resolved]["encoder"] in have:
            names.append(name)

    if DEFAULT_PRESET in names:
        names.remove(DEFAULT_PRESET)
        names.insert(0, DEFAULT_PRESET)
    return names or [DEFAULT_PRESET]


def ffprobe_path():
    """ffprobe sitting beside the ffmpeg we already found, if there is one.

    imageio-ffmpeg ships ffmpeg WITHOUT ffprobe, which is common in
    ComfyUI installs -- so callers must handle None.
    """
    import os
    exe = ffmpeg_path()
    base = os.path.basename(exe)
    for a, b in (("ffmpeg.exe", "ffprobe.exe"), ("ffmpeg", "ffprobe")):
        if base == a:
            cand = os.path.join(os.path.dirname(exe), b)
            if os.path.isfile(cand):
                return cand
    w = shutil.which("ffprobe")
    return w or None


def probe_video(path):
    """What actually got encoded: codec, profile, pixel format.

    4:4:4 and bit depth are properties of the stream, not container tags,
    so they never show up in a player's metadata panel. This reads them
    back off the finished file so the node can state them plainly.

    Returns {} rather than raising -- this is a nicety, never a reason to
    fail a render that already succeeded.
    """
    import re

    probe = ffprobe_path()
    if probe:
        try:
            out = subprocess.run(
                [probe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=codec_name,profile,pix_fmt",
                 "-of", "default=nw=1", path],
                capture_output=True, text=True, timeout=15,
            ).stdout
            got = {}
            for line in out.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    if v.strip() and v.strip() != "unknown":
                        got[k.strip()] = v.strip()
            if got:
                return {"codec": got.get("codec_name", ""),
                        "profile": got.get("profile", ""),
                        "pix_fmt": got.get("pix_fmt", "")}
        except Exception:
            pass

    # no ffprobe: ffmpeg -i prints the stream line to stderr and exits
    # non-zero because no output was given. That is expected, not an error.
    try:
        err = subprocess.run([ffmpeg_path(), "-hide_banner", "-i", path],
                             capture_output=True, text=True, timeout=15).stderr
        m = re.search(r"Video:\s*([^\s,(]+)(.*)", err)
        if not m:
            return {}
        rest = m.group(2)
        pix = re.search(r",\s*((?:yuv|gbr|rgb|bgr)[a-z0-9]*)", rest)
        prof = re.search(r"\(([^)]*(?:4:4:4|4:2:2|4:2:0|Main|High|Predictive)[^)]*)\)", rest)
        return {"codec": m.group(1),
                "profile": (prof.group(1).strip() if prof else ""),
                "pix_fmt": (pix.group(1) if pix else "")}
    except Exception:
        return {}


def pipe_format(name):
    """Raw pixel format the node should write to ffmpeg's stdin.

    ComfyUI IMAGE is float. Quantising it to 8-bit rgb24 on the way in
    caps everything downstream at 8 bits -- including the presets that
    claim 10-bit or lossless. High-fidelity presets ask for rgb48le so
    the depth they advertise is actually fed.
    """
    _, p = resolve_preset(name)
    return p.get("pipe", "rgb24")


def container_for(name):
    _, p = resolve_preset(name)
    return p["container"]


def needs_preview_copy(name):
    """True if the master won't play in a browser and the node should
    also emit a small H.264 file for the in-node preview."""
    _, p = resolve_preset(name)
    return not p["browser_playable"]


# ---------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------

def write_ffmetadata(path, fields):
    """Write an ffmetadata file.

    Metadata goes in via a file rather than -metadata on the command
    line because a ComfyUI workflow JSON is easily tens of kilobytes and
    Windows caps a command line at about 32k. This has no length limit.
    """
    def esc(s):
        s = str(s)
        for ch in ("\\", "=", ";", "#"):
            s = s.replace(ch, "\\" + ch)
        return s.replace("\n", "\\\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(";FFMETADATA1\n")
        for k, v in fields.items():
            if v:
                f.write("%s=%s\n" % (k, esc(v)))
    return path


def build_ffmpeg_cmd(preset, width, height, fps, out_path,
                     audio_path=None, audio_rate=None, audio_channels=2,
                     preview_path=None, meta_path=None):
    """Build a single-pass ffmpeg command.

    Frames are written to stdin as raw rgb24, or rgb48le for the
    high-fidelity presets -- see pipe_format(). Audio, if given, is a
    second input. One process, one master file -- nothing to clean up
    afterwards.

    If preview_path is given, a second small H.264 output is muxed in
    the same pass. Use it for ProRes/FFV1 masters, which browsers can't
    decode. It costs one extra output stream, not another process.
    """
    name, p = resolve_preset(preset)

    cmd = [
        ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo",
        "-pix_fmt", p.get("pipe", "rgb24"),
        "-s", "%dx%d" % (int(width), int(height)),
        "-r", "%.6f" % float(fps),
        "-i", "-",
    ]

    if audio_path:
        if audio_rate:
            # raw 32-bit float PCM -- avoids a WAV library and keeps
            # full precision from ComfyUI's float audio tensors
            cmd += ["-f", "f32le",
                    "-ar", str(int(audio_rate)),
                    "-ac", str(int(audio_channels))]
        cmd += ["-i", audio_path]

    meta_idx = None
    if meta_path:
        meta_idx = 2 if audio_path else 1
        cmd += ["-f", "ffmetadata", "-i", meta_path]

    # ---- master output ----
    cmd += ["-map", "0:v:0"]
    if audio_path:
        cmd += ["-map", "1:a:0"] + AUDIO_ARGS + ["-shortest"]
    if meta_idx is not None:
        cmd += ["-map_metadata", str(meta_idx)]
    cmd += list(p["args"])
    if p["container"] == "mp4":
        cmd += ["-movflags", "+faststart+use_metadata_tags"]
    cmd += [out_path]

    # ---- optional browser-playable preview, same pass ----
    if preview_path:
        cmd += ["-map", "0:v:0"]
        if audio_path:
            cmd += ["-map", "1:a:0", "-c:a", "aac", "-b:a", "128k", "-shortest"]
        cmd += [
            "-c:v", "libx264",
            "-crf", "26",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            preview_path,
        ]

    return cmd


if __name__ == "__main__":
    print("ffmpeg:", ffmpeg_path())
    print("available:", available_presets())
    for n in available_presets():
        r, p = resolve_preset(n)
        mark = "" if r == n else "  -> falls back to %s" % r
        print("  %-28s %s%s" % (n, p["note"], mark))
