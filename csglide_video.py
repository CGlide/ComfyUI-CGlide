"""
csglide_video.py -- Glide Video

Encodes an IMAGE batch (plus optional AUDIO) into exactly one video file.

Why this exists: VHS_VideoCombine encodes a silent file, then remuxes
audio into a second file, then writes a PNG for workflow metadata --
three files when you wanted one. Glide Video pipes frames straight into
a single ffmpeg process with audio as a second input, so there is
nothing to duplicate and nothing to prune.

Drop this next to csglide_video_presets.py and register it in __init__.py.
"""

import collections
import json
import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np

try:
    import folder_paths
except ImportError:  # allows importing outside ComfyUI for testing
    folder_paths = None

try:
    from .csglide_video_presets import (
        available_presets, build_ffmpeg_cmd, container_for,
        needs_preview_copy, pipe_format, probe_video, resolve_preset,
        CONTAINER_CHOICES, containers_for, resolve_container,
        write_ffmetadata, PRESETS, DEFAULT_PRESET,
    )
except ImportError:
    from csglide_video_presets import (
        available_presets, build_ffmpeg_cmd, container_for,
        needs_preview_copy, pipe_format, probe_video, resolve_preset,
        CONTAINER_CHOICES, containers_for, resolve_container,
        write_ffmetadata, PRESETS, DEFAULT_PRESET,
    )


# How many lines of ffmpeg stderr to keep. The interesting line is always
# the LAST one -- x265 and friends print a long configuration banner on
# startup, so keeping the head tells you only that the encoder launched.
STDERR_LINES = 400


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------

def _output_dir():
    if folder_paths is not None:
        return folder_paths.get_output_directory()
    return os.path.abspath("output")


def _temp_dir():
    if folder_paths is not None:
        return folder_paths.get_temp_directory()
    return tempfile.gettempdir()


def _next_path(directory, prefix, ext):
    """ComfyUI-style incrementing filename: prefix_00001.ext"""
    subdir = os.path.dirname(prefix)
    base = os.path.basename(prefix) or "glide"
    target = os.path.join(directory, subdir) if subdir else directory
    os.makedirs(target, exist_ok=True)

    n = 1
    for name in os.listdir(target):
        if name.startswith(base + "_"):
            stem = os.path.splitext(name)[0]
            tail = stem[len(base) + 1:].split("_")[0]
            if tail.isdigit():
                n = max(n, int(tail) + 1)

    filename = "%s_%05d.%s" % (base, n, ext)
    return os.path.join(target, filename), filename, subdir


def _free_space_mb(path):
    """Free megabytes on the volume holding path, or None if unknown."""
    try:
        probe = path
        while probe and not os.path.isdir(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        return shutil.disk_usage(probe).free // (1024 * 1024)
    except Exception:
        return None


def _write_audio_pcm(audio, path):
    """Write ComfyUI AUDIO to raw 32-bit float PCM.

    ComfyUI AUDIO is {"waveform": tensor [B, C, T], "sample_rate": int}.
    Raw f32le keeps full precision and needs no WAV library -- ffmpeg is
    told the format on the command line.

    Returns (sample_rate, channels) or None.
    """
    if not audio:
        return None
    wf = audio.get("waveform")
    sr = int(audio.get("sample_rate", 48000))
    if wf is None:
        return None

    arr = wf.detach().cpu().numpy() if hasattr(wf, "detach") else np.asarray(wf)
    if arr.ndim == 3:
        arr = arr[0]          # first item of batch -> [C, T]
    if arr.ndim == 1:
        arr = arr[None, :]

    channels = int(arr.shape[0])
    # ffmpeg wants interleaved samples: [C, T] -> [T, C]
    interleaved = np.ascontiguousarray(arr.T.astype(np.float32))
    with open(path, "wb") as f:
        f.write(interleaved.tobytes())
    return sr, channels


def _frame_bytes(frame, deep):
    """One ComfyUI IMAGE frame -> raw bytes for the ffmpeg pipe."""
    a = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
    # ComfyUI IMAGE is float 0..1, shape [H, W, C]
    if a.shape[2] == 4:
        a = a[:, :, :3]
    if deep:
        # 16-bit little-endian: the 10-bit and lossless presets are only
        # as good as what reaches them, and rgb24 would truncate to 8 bits
        # before ffmpeg ever sees the frame
        b = np.clip(a * 65535.0 + 0.5, 0, 65535).astype("<u2")
    else:
        b = np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return b.tobytes()


def _drain(stream, sink):
    """Consume a pipe line by line into a bounded deque.

    This has to run on its own thread. ffmpeg writes progress and warnings
    to stderr while it encodes; if nobody reads that pipe it fills (64 KB
    on Windows), ffmpeg blocks writing to it, stops reading stdin, and the
    frame loop blocks writing to stdin. Reading stderr only after the last
    frame -- the way this node used to -- is a deadlock waiting for a
    chatty encoder, and it truncates the error when one does appear.
    """
    try:
        for line in iter(stream.readline, b""):
            sink.append(line.decode("utf-8", "replace").rstrip("\r\n"))
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _encode_once(cmd, images, deep):
    """Run one ffmpeg pass. Returns (returncode, stderr_tail, broken_at).

    broken_at is the frame index where the pipe died, or None if every
    frame was written. A broken pipe means ffmpeg quit early -- the reason
    is in the stderr tail, not in the exception.
    """
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    lines = collections.deque(maxlen=STDERR_LINES)
    pump = threading.Thread(target=_drain, args=(proc.stderr, lines), daemon=True)
    pump.start()

    broken_at = None
    try:
        for i, frame in enumerate(images):
            try:
                proc.stdin.write(_frame_bytes(frame, deep))
            except (BrokenPipeError, OSError):
                broken_at = i
                break
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
    return code, "\n".join(lines), broken_at


def _fallback_chain(failed):
    """Preset names to try after `failed`, most compatible first.

    Point of this: by the time encoding starts, the expensive part of the
    graph is already done. Losing a finished sampling run because one
    encoder refused is the worst possible failure, so trade the codec
    rather than the render.
    """
    names = available_presets()
    chain = []

    def add(name):
        if name and name in names and name != failed and name not in chain:
            chain.append(name)

    add(DEFAULT_PRESET)
    for want in ("H.264", "FFV1"):
        for name in names:
            if want.lower() in name.lower() and "4:4:4" not in name:
                add(name)
                break
    for name in names:
        add(name)
    return chain


# ---------------------------------------------------------------------
# node
# ---------------------------------------------------------------------

class CSGlideVideo:
    """Glide Video -- one clip in, one file out."""

    # ---------------- chain note ----------------

    CHAIN_TAG = "description"

    # Widget names worth recording, and what to call them in the note. Keyed on
    # the widget rather than the node class on purpose: class names change with
    # every pack and every ComfyUI release, but a seed has been called "seed" or
    # "noise_seed" for as long as any of this has existed.
    _CHAIN_KEYS = (
        (("seed", "noise_seed"), "seed"),
        (("steps",), "steps"),
        (("lora_name",), "lora"),
        (("unet_name", "ckpt_name", "model_name"), "model"),
        (("sampler_name",), "sampler"),
    )

    # Substrings marking a node as an accelerator or precision trick. Those
    # carry no useful widget - what matters is that one was in the graph at
    # all, because it changes the result and is the first thing you forget
    # having enabled.
    _CHAIN_ACCEL = ("sage", "attention", "quant", "gguf", "compile",
                    "teacache", "nunchaku", "fp8", "torchao")

    @classmethod
    def _summarise(cls, prompt, filename):
        """One line saying how this clip was made.

        Deliberately not the whole workflow JSON: that is already in the
        "comment" tag, and carrying a copy per link would put three workflows
        in a three clip chain. This is the short answer to "what made this
        bit", which is the question actually asked six weeks later.
        """
        found, accel = {}, []
        for _, node in sorted((prompt or {}).items()):
            if not isinstance(node, dict):
                continue
            cls_name = str(node.get("class_type", ""))
            low = cls_name.lower()
            if any(w in low for w in cls._CHAIN_ACCEL) and cls_name not in accel:
                accel.append(cls_name)
            for key, value in (node.get("inputs") or {}).items():
                # A list here is a link to another node, not a set value.
                if isinstance(value, (list, dict)) or value is None:
                    continue
                for names, label in cls._CHAIN_KEYS:
                    if str(key).lower() in names:
                        text = str(value)
                        if isinstance(value, float) and value.is_integer():
                            text = str(int(value))
                        found.setdefault(label, [])
                        if text not in found[label]:
                            found[label].append(text)
        bits = ["file=%s" % filename]
        for _, label in cls._CHAIN_KEYS:
            if found.get(label):
                bits.append("%s=%s" % (label, ",".join(found[label])))
        if accel:
            bits.append("accel=%s" % ",".join(accel))
        return "  ".join(bits)

    @classmethod
    def INPUT_TYPES(cls):
        presets = available_presets()
        lines = []
        for name in presets:
            _, p = resolve_preset(name)
            lines.append("%s -- %s" % (name, p["note"]))
        preset_tip = "\n".join(lines)

        clines = ["auto -- whatever the preset was designed for."]
        for name in presets:
            clines.append("%s -- %s" % (name, ", ".join(containers_for(name))))
        container_tip = ("Wrapper for the encoded stream. The codec is "
                         "unchanged, so this only affects what will open the "
                         "file, not how it looks.\n"
                         + "\n".join(clines)
                         + "\nAsking for one a preset cannot mux keeps the "
                           "preset's own container and says so in the log.")

        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Playback frame rate. MiniMax H3 is 24. "
                               "Setting this wrong changes playback speed, "
                               "it does not drop or add frames.",
                }),
                "preset": (presets, {
                    "default": presets[0],
                    "tooltip": preset_tip,
                }),
                "filename_prefix": ("STRING", {
                    "default": "glide/GlideVideo",
                    "tooltip": "Path under the output folder. A slash makes a "
                               "subfolder. A counter is appended automatically.",
                }),
                "save_output": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On: writes to the output folder. "
                               "Off: writes to temp, so previews do not pile up.",
                }),
                "save_metadata": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Embed the prompt and workflow in the file's "
                               "comment field. Readable with exiftool or VLC. "
                               "Drag-and-drop restore into ComfyUI is reliable "
                               "for MKV, hit-or-miss for MP4 depending on "
                               "frontend version.",
                }),
                "fallback_on_failure": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If the chosen encoder fails, retry with a more "
                               "compatible preset instead of throwing away a "
                               "finished render. The substitution is printed "
                               "loudly and the real stream is shown in the "
                               "preview meta line.",
                }),
                # Appended LAST on purpose. ComfyUI restores widget values
                # by position, so a new widget inserted higher up would
                # shift every value in every saved workflow. At the end,
                # older workflows simply have no value for it and take the
                # default.
                "container": (CONTAINER_CHOICES, {
                    "default": "auto",
                    "tooltip": container_tip,
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "From VAEDecodeAudio. Muxed in the same pass, "
                               "so there is no second file.",
                }),
                "chain": ("STRING", {
                    "forceInput": True,
                    "tooltip": "The chain note out of Glide Join. Wire it and a "
                               "joined clip records how EVERY clip in it was "
                               "made - seeds, steps, LoRAs, accelerators - so a "
                               "finished 40 second file can still tell you which "
                               "seed produced the middle of it.",
                }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    OUTPUT_NODE = True
    FUNCTION = "combine"
    CATEGORY = "CGlide"
    DESCRIPTION = (
        "Encode frames and audio into a single video file. "
        "No silent duplicate, no metadata PNG."
    )

    def combine(self, images, fps, preset, filename_prefix,
                save_output=True, save_metadata=True, fallback_on_failure=True,
                container="auto", audio=None, chain=None, prompt=None,
                extra_pnginfo=None):
        if images is None or len(images) == 0:
            raise ValueError("Glide Video: no frames on the images input.")

        directory = _output_dir() if save_output else _temp_dir()

        # -- frame geometry from the first frame -------------------------
        first = images[0]
        arr0 = first.detach().cpu().numpy() if hasattr(first, "detach") else np.asarray(first)
        height, width = int(arr0.shape[0]), int(arr0.shape[1])

        uid = "%d_%d" % (os.getpid(), id(self))

        # -- audio -------------------------------------------------------
        pcm_path = None
        audio_rate = None
        audio_channels = 2
        if audio is not None:
            pcm_path = os.path.join(_temp_dir(), "glide_audio_%s.pcm" % uid)
            os.makedirs(os.path.dirname(pcm_path), exist_ok=True)
            info = _write_audio_pcm(audio, pcm_path)
            if info is None:
                pcm_path = None
            else:
                audio_rate, audio_channels = info

        # -- metadata ----------------------------------------------------
        meta_path = None
        if save_metadata and (prompt is not None or extra_pnginfo):
            payload = {}
            if prompt is not None:
                payload["prompt"] = prompt
            if extra_pnginfo:
                payload.update(extra_pnginfo)   # includes "workflow"
            meta_path = os.path.join(_temp_dir(), "glide_meta_%s.txt" % uid)
            os.makedirs(os.path.dirname(meta_path), exist_ok=True)
            # Written for real inside attempt(), once the output filename is
            # known - the chain note names the file it describes, and a
            # fallback to another preset can change that name.
            meta_fields = {
                "comment": json.dumps(payload, separators=(",", ":")),
                "encoder": "Glide Video",
            }
            write_ffmetadata(meta_path, meta_fields)

        deep_cache = {}

        def attempt(name):
            """Encode with one preset. Returns a dict on success, else None."""
            resolved_name, resolved = resolve_preset(name)
            ext, honoured = resolve_container(name, container)
            if not honoured:
                print("[Glide Video] '%s' cannot be muxed into .%s -- writing "
                      ".%s instead (this preset accepts: %s)"
                      % (name, str(container).lower(), ext,
                         ", ".join(containers_for(name))))
            out_path, filename, subdir = _next_path(directory, filename_prefix, ext)

            # Append this clip to the history the source clip brought with it.
            # Nothing is wired for an ordinary render, so `chain` is None and
            # this is simply the first line of a history that may never grow.
            if meta_path:
                note = self._summarise(prompt, filename)
                previous = (chain or "").strip()
                write_ffmetadata(meta_path, dict(
                    meta_fields,
                    **{self.CHAIN_TAG: (previous + "\n" + note) if previous else note}))

            preview_path = None
            preview_name = None
            if needs_preview_copy(name):
                preview_name = os.path.splitext(filename)[0] + "_preview.mp4"
                preview_path = os.path.join(_temp_dir(), preview_name)

            cmd = build_ffmpeg_cmd(
                preset=name,
                width=width, height=height, fps=fps,
                out_path=out_path,
                audio_path=pcm_path,
                audio_rate=audio_rate,
                audio_channels=audio_channels,
                preview_path=preview_path,
                meta_path=meta_path,
                container=ext,
            )

            deep = deep_cache.setdefault(name, pipe_format(name) == "rgb48le")
            code, err, broken_at = _encode_once(cmd, images, deep)

            if code != 0:
                where = ("pipe closed at frame %d of %d"
                         % (broken_at, len(images))) if broken_at is not None \
                        else "all %d frames written" % len(images)
                free = _free_space_mb(out_path)
                disk = "unknown" if free is None else "%d MB free" % free
                report = (
                    "Glide Video: ffmpeg failed\n"
                    "  preset   : %s (encoder %s)\n"
                    "  exit code: %s\n"
                    "  frames   : %s\n"
                    "  target   : %s (%s)\n"
                    "  ffmpeg   : %s\n"
                    "--- last lines of ffmpeg stderr ---\n%s"
                    % (name, resolved["encoder"], code, where, out_path, disk,
                       " ".join(str(c) for c in cmd), err.strip() or "(nothing captured)")
                )
                if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
                    try:
                        os.remove(out_path)
                    except OSError:
                        pass
                return {"error": report}

            if resolved_name != name:
                print("[Glide Video] '%s' unavailable, used '%s'" % (name, resolved_name))

            # Read the finished file back. Chroma sampling and bit depth are
            # stream properties, not container tags, so no player metadata
            # panel will ever show them -- this is the only honest answer.
            info = probe_video(out_path)
            stream = " / ".join(v for v in (info.get("codec"), info.get("profile"),
                                            info.get("pix_fmt")) if v)
            if stream:
                print("[Glide Video] %s  ->  %s" % (os.path.basename(out_path), stream))

            return {
                "out_path": out_path, "filename": filename, "subdir": subdir,
                "preview_path": preview_path, "preview_name": preview_name,
                "ext": ext, "stream": stream,
            }

        try:
            first_error = None
            result = attempt(preset)
            if "error" in result:
                first_error = result["error"]
                if not fallback_on_failure:
                    raise RuntimeError(first_error)

                print("[Glide Video] " + first_error)
                result = None
                for alt in _fallback_chain(preset):
                    print("[Glide Video] retrying with fallback preset '%s' "
                          "so the render is not lost" % alt)
                    trial = attempt(alt)
                    if "error" not in trial:
                        print("[Glide Video] !! saved with FALLBACK preset '%s' "
                              "instead of '%s' -- see the error above" % (alt, preset))
                        result = trial
                        break
                    print("[Glide Video] fallback '%s' also failed" % alt)

                if result is None:
                    raise RuntimeError(
                        first_error
                        + "\n\nEvery fallback preset failed too. When nothing "
                          "encodes, the cause is usually outside the codec: "
                          "no free space on the output volume, the output file "
                          "locked by a player or indexer, or a broken ffmpeg."
                    )
        finally:
            for tmp in (pcm_path, meta_path):
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        # -- UI payload --------------------------------------------------
        ui_name = result["preview_name"] or result["filename"]
        ui_type = "temp" if (result["preview_path"] or not save_output) else "output"
        ui_sub = "" if result["preview_path"] else result["subdir"]

        return {
            "ui": {
                "glide_video": [{
                    "filename": ui_name,
                    "subfolder": ui_sub,
                    "type": ui_type,
                    "fps": float(fps),
                    "frames": int(len(images)),
                    # Sent from here rather than read off the <video>
                    # element: the preview copy is what actually loads for
                    # the masters, and for a format the browser cannot
                    # decode at all there is no element to ask.
                    "width": int(width),
                    "height": int(height),
                    "format": "video/mp4" if ui_name.endswith(".mp4") else "video/" + result["ext"],
                    "master": os.path.basename(result["out_path"]),
                    "stream": result["stream"],
                }],
            },
            "result": (result["out_path"],),
        }


NODE_CLASS_MAPPINGS = {"CSGlideVideoCS": CSGlideVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideVideoCS": "Glide Video"}
