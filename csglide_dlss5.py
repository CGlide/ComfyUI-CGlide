"""Glide DLSS5 - drop a video on the node, scrub it, preview DLSS 5 at the playhead.

The engine is Blueforcer's ComfyUI-DLSS5-Enhancer (MIT), a separate pack the
user installs. This node ships none of it: it uses the modules that pack
already loaded, so a DLSS5 update reaches this node without touching it.

Two paths, deliberately separate:

- PREVIEW never goes through the queue. The button posts the connected DLSS5
  Settings node's widget values to a server route, which builds the exact
  same config the Settings node would (DlssOptions.create + SessionConfig),
  renders a short burst of frames ending at the playhead and returns a
  before/after pair. Queueing a partial prompt was the alternative, but on a
  ComfyUI that ignores partial targets it would run every output node in the
  graph - an H3 render included - just to look at eight frames.
- RENDER is the queue. The node's execute hands the whole file to the pack's
  own DLSS5 Enhance Video File logic, so the output is byte-for-byte what that
  node would have produced: same encoder, same audio/metadata mux, same
  feature-18 verification.

Why a burst and not one frame: DLSS accumulates over consecutive frames, and
frame one of any session has no history. The last frame of a short run is
what that frame looks like inside a full render.
"""
from __future__ import annotations

import asyncio
import io as _io
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import folder_paths

try:
    import av
except ImportError:  # the DLSS5 pack requires it too; its own error is clearer
    av = None

try:
    from PIL import Image
except ImportError:
    Image = None

import numpy as np
from aiohttp import web
from server import PromptServer

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v", ".mts", ".m2ts", ".ts"}
UPLOAD_SUBFOLDER = "glide_dlss5"
PREVIEW_SUBFOLDER = "glide_dlss5"

# One DLSS worker at a time from this node: two previews racing each other
# would only fight over the same GPU and the second would lose.
_PREVIEW_LOCK = threading.Lock()


# --------------------------------------------------------------- the engine

class _Engine:
    """The DLSS5 pack's modules, as that pack already loaded them.

    ComfyUI names a custom-node package after its folder - sometimes the full
    folder path - so the package has no fixed name. What is fixed is its
    layout: <pack>.dlss5.session defines DlssSession. Find that, and every
    sibling module follows from the same root.

    Nothing is imported here. The DLSS5 pack imports every module this node
    uses when it loads its own nodes, so they are all in sys.modules already;
    reading them from there needs no import call with a computed name - the
    pattern the registry scanner files under obfuscated code."""

    _root: str | None = None

    @classmethod
    def root(cls) -> str:
        if cls._root and cls._root + ".dlss5.session" in sys.modules:
            return cls._root
        for name, module in list(sys.modules.items()):
            if name.endswith(".dlss5.session") and hasattr(module, "DlssSession"):
                cls._root = name[: -len(".dlss5.session")]
                return cls._root
        raise RuntimeError(
            "Glide DLSS5 needs ComfyUI-DLSS5-Enhancer installed and loaded "
            "(it provides the DLSS 5 engine and the DLSS5 Settings node)."
        )

    @classmethod
    def mod(cls, relative: str):
        module = sys.modules.get(cls.root() + "." + relative)
        if module is None:
            raise RuntimeError(
                "ComfyUI-DLSS5-Enhancer is loaded but has no " + relative
                + " module - its version does not match what Glide DLSS5 expects."
            )
        return module


def _choices(attr: str, fallback: list[str]) -> list[str]:
    try:
        return list(getattr(_Engine.mod("dlss5.media"), attr))
    except Exception:
        return fallback


# ---------------------------------------------------------------- sources

def _allowed_roots() -> list[Path]:
    return [Path(folder_paths.get_input_directory()).resolve(),
            Path(folder_paths.get_output_directory()).resolve(),
            Path(folder_paths.get_temp_directory()).resolve()]


def _resolve_source(value: str) -> Path:
    """A video inside ComfyUI's input, output or temp folder - nowhere else.

    The routes below take this value straight from an HTTP request, so it is
    confined the same way the pack's other file routes are: relative names
    live in the input folder, absolute paths are accepted only when they
    resolve (symlinks and '..' included) inside one of ComfyUI's own folders.
    A dropped file lands in input/glide_dlss5; a render of yours is already
    in output. Anything else on the disk is refused, not read."""
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("No video yet. Drop a video file on the node.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(folder_paths.get_input_directory()) / raw
    candidate = candidate.resolve()
    if not any(candidate.is_relative_to(root) for root in _allowed_roots()):
        raise ValueError("Glide DLSS5 only reads videos inside ComfyUI's input, "
                         "output or temp folders. Drop the file on the node instead.")
    if candidate.suffix.lower() not in VIDEO_EXTS:
        raise ValueError(f"{candidate.name} is not a video file this node reads.")
    if not candidate.is_file():
        raise FileNotFoundError(f"No video file at {candidate}")
    return candidate


def _rotation(stream) -> int:
    """Container rotation, the same way the pack reads it, so previews match."""
    rotation = 0
    try:
        tag = (stream.metadata or {}).get("rotate")
        if tag:
            rotation = int(float(tag))
    except (TypeError, ValueError):
        pass
    try:
        for side in getattr(stream, "side_data", None) or []:
            value = getattr(side, "rotation", None)
            if value is not None:
                rotation = int(value)
    except Exception:
        pass
    return rotation % 360


def _info(path: Path) -> dict:
    if av is None:
        raise RuntimeError("PyAV is not installed; the DLSS5 pack needs it as well.")
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 24)
        frames = int(stream.frames or 0)
        duration = 0.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
        if frames <= 0:
            frames = max(1, int(round(duration * fps)))
        rotation = _rotation(stream)
        width, height = stream.codec_context.width, stream.codec_context.height
        if rotation in (90, 270):
            width, height = height, width
        return {"frames": frames, "fps": fps, "duration": duration,
                "width": width, "height": height, "rotation": rotation,
                "pix_fmt": stream.codec_context.format.name if stream.codec_context.format else ""}
    finally:
        container.close()


def _decode(path: Path, start: int, count: int) -> list:
    """RGBA frames [start, start+count), by seeking - not decoding from zero.

    Index is derived from each frame's own timestamp, so a variable frame rate
    or a stream that does not start at 0 still lands on the right frame."""
    rotate = _Engine.mod("dlss5.media").rotate_frame
    container = av.open(str(path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        fps = float(stream.average_rate or stream.guessed_rate or 24)
        base = stream.time_base
        first = stream.start_time or 0
        rotation = _rotation(stream)
        if start > 0:
            container.seek(first + int(start / fps / float(base)), stream=stream,
                           backward=True, any_frame=False)
        frames = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            index = int(round((frame.pts - first) * float(base) * fps))
            if index < start:
                continue
            frames.append(rotate(frame.to_ndarray(format="rgba"), rotation))
            if len(frames) >= count:
                break
        if not frames:
            raise RuntimeError(f"Could not decode frame {start} of {path.name}.")
        return frames
    finally:
        container.close()


def _save_png(rgba, subfolder: str, stem: str) -> dict:
    directory = Path(folder_paths.get_temp_directory()) / subfolder
    directory.mkdir(parents=True, exist_ok=True)
    name = f"{stem}_{uuid.uuid4().hex[:10]}.png"
    Image.fromarray(rgba[..., :3]).save(directory / name, compress_level=1)
    return {"filename": name, "subfolder": subfolder, "type": "temp"}


# ---------------------------------------------------------------- preview

def _settings_from_widgets(values: dict):
    """Exactly what the DLSS5 Settings node's execute builds from its widgets."""
    settings = _Engine.mod("dlss5.settings")
    options = settings.DlssOptions.create(
        upscaling_mode=values["upscaling_mode"],
        nr_preset=values["nr_preset"],
        nr_style=values["nr_style"],
        dlss_model_preset=values["dlss_model_preset"],
        nr_intensity=float(values["nr_intensity"]),
        local_tone_strength=float(values["local_tone_strength"]),
        local_structure_strength=float(values["local_structure_strength"]),
        skin_structure_strength=float(values["skin_structure_strength"]),
        automatic_mask=bool(values["automatic_mask"]),
        warmup_frames=int(values.get("warmup_frames", 0)),
        motion_mode=values.get("motion", "auto"),
        scene_change_threshold=float(values.get("scene_change_threshold", 0.24)),
    )
    return settings.SessionConfig(options=options, runtime_dir=str(values.get("runtime_dir", "") or ""))


def run_preview(file: str, frame: int, count: int, widgets: dict) -> dict:
    source = _resolve_source(file)
    config = _settings_from_widgets(widgets)
    common = _Engine.mod("nodes.common")
    imaging = _Engine.mod("dlss5.imaging")
    motion_mod = _Engine.mod("dlss5.motion")
    DlssSession = _Engine.mod("dlss5.session").DlssSession

    info = _info(source)
    frame = max(0, min(int(frame), info["frames"] - 1))
    count = max(1, min(int(count), 32))
    start = max(0, frame - count + 1)
    frames = _decode(source, start, frame - start + 1)
    layout = common.resolve_layout(config)
    options = config.options
    height, width = frames[0].shape[:2]

    started = time.perf_counter()
    last = None
    with DlssSession(layout, options, input_width=width, input_height=height,
                     frame_count=len(frames)) as session:
        guide = motion_mod.TemporalGuide(
            session.render_width, session.render_height,
            flow_width=options.flow_width,
            scene_change_threshold=options.scene_change_threshold,
            enabled=options.wants_motion(len(frames)),
        )
        for index, rgba in enumerate(frames):
            fitted = imaging.fit_frame(rgba, session.render_width, session.render_height)
            step = guide.process(fitted)
            last, _pts = session.submit(index=index, rgba=fitted, motion=step.motion,
                                        reset=step.reset, pts=index)
        out_w, out_h = session.output_width, session.output_height
    elapsed = time.perf_counter() - started

    # The "before" is the same source frame at the output size, so the two
    # halves of the slider line up pixel for pixel. Lanczos: the honest
    # plain-upscale comparison, which is exactly what DLSS is being judged on.
    before = Image.fromarray(frames[-1][..., :3]).resize((out_w, out_h), Image.LANCZOS)
    return {
        "before": _save_png(np.asarray(before.convert("RGBA")), PREVIEW_SUBFOLDER, "before"),
        "after": _save_png(last, PREVIEW_SUBFOLDER, "after"),
        "frame": frame, "burst": len(frames), "seconds": round(elapsed, 2),
        "input": [width, height], "output": [out_w, out_h],
    }


# ----------------------------------------------------------------- routes

routes = PromptServer.instance.routes


@routes.get("/csglide_dlss5/info")
async def _route_info(request):
    try:
        path = _resolve_source(request.query.get("file", ""))
        return web.json_response(await asyncio.to_thread(_info, path))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@routes.get("/csglide_dlss5/frame")
async def _route_frame(request):
    """One source frame as a JPEG, for the scrub bar. Any codec the pack can
    decode works - the browser never has to play the file itself, which it
    could not for 4:4:4 HEVC in MKV anyway."""
    try:
        path = _resolve_source(request.query.get("file", ""))
        index = int(request.query.get("index", "0"))
        width = max(64, min(int(request.query.get("w", "960")), 2048))

        def grab():
            rgba = _decode(path, max(0, index), 1)[0]
            image = Image.fromarray(rgba[..., :3])
            if image.width > width:
                image = image.resize((width, max(1, round(image.height * width / image.width))),
                                     Image.BILINEAR)
            buffer = _io.BytesIO()
            image.save(buffer, format="JPEG", quality=88)
            return buffer.getvalue()

        data = await asyncio.to_thread(grab)
        return web.Response(body=data, content_type="image/jpeg",
                            headers={"Cache-Control": "no-store"})
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=400)


@routes.post("/csglide_dlss5/preview")
async def _route_preview(request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "Bad request."}, status=400)
    if not _PREVIEW_LOCK.acquire(blocking=False):
        return web.json_response({"error": "A preview is already running."}, status=409)
    try:
        result = await asyncio.to_thread(
            run_preview, body.get("file", ""), int(body.get("frame", 0)),
            int(body.get("count", 8)), dict(body.get("settings") or {}),
        )
        return web.json_response(result)
    except KeyError as exc:
        return web.json_response(
            {"error": f"The DLSS5 Settings node is missing {exc}. Connect a DLSS5 Settings node directly."},
            status=400)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    finally:
        _PREVIEW_LOCK.release()


# ------------------------------------------------------------------- naming

def _rename_like_source(rendered: Path, source: Path, suffix: str) -> str:
    """Give the rendered file the source video's name plus a suffix, in the
    folder the engine wrote it to. Never overwrites: an existing name gets
    _2, _3... If anything goes wrong the engine's own name is kept."""
    try:
        if not rendered.is_file():
            return str(rendered)
        tag = "".join(c for c in (suffix or "").strip() if c not in '<>:"/\\|?*')
        stem = source.stem + (("_" + tag) if tag else "")
        target = rendered.with_name(stem + rendered.suffix)
        if target == rendered:
            return str(rendered)
        n = 2
        while target.exists():
            target = rendered.with_name(f"{stem}_{n}{rendered.suffix}")
            n += 1
        os.replace(rendered, target)
        return str(target)
    except Exception as error:
        print(f"[Glide DLSS5] kept the engine's file name, rename failed: {error}")
        return str(rendered)


# ------------------------------------------------------------------- node

class CSGlideDLSS5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("DLSS5_SETTINGS", {"tooltip": "Connect a DLSS5 Settings node. Preview reads its current values."}),
                "video": ("STRING", {"default": "", "tooltip": "Drop a video on the node. A pasted path must be inside ComfyUI's input, output or temp folder."}),
                "codec": (_choices("CODECS", ["HEVC", "H.264", "AV1", "ProRes Proxy"]), {"default": "HEVC"}),
                "container": (_choices("CONTAINERS", ["MKV", "MP4", "MOV"]), {"default": "MKV"}),
                "quality": (_choices("QUALITIES", ["Auto", "Good", "Best", "Max"]), {"default": "Auto"}),
                "filename_prefix": ("STRING", {"default": "GlideDLSS5"}),
                "copy_audio": ("BOOLEAN", {"default": True}),
                "preview_frames": ("INT", {"default": 8, "min": 1, "max": 32,
                                           "tooltip": "Frames rendered up to the playhead for a preview. DLSS builds up over frames; 8 is a fair look."}),
            },
            "optional": {
                "output_directory": ("STRING", {"default": "", "tooltip": "Empty = ComfyUI output folder."}),
                "name_from_source": ("BOOLEAN", {"default": True,
                                     "tooltip": "On: the output keeps the source video's name, with filename_prefix added as a suffix "
                                                "(01_shot_00001.mp4 -> 01_shot_00001_DLSS5.mkv). Off: prefix + timestamp, as before."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video_path", "frames")
    FUNCTION = "render"
    CATEGORY = "CGlide"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, video="", **kwargs):
        try:
            status = _resolve_source(video).stat()
            return f"{status.st_mtime_ns}:{status.st_size}"
        except Exception:
            return float("nan")

    def render(self, settings, video, codec, container, quality, filename_prefix,
               copy_audio, preview_frames, output_directory="", name_from_source=True):
        source = _resolve_source(video)
        node = _Engine.mod("nodes.enhance_video").DLSS5EnhanceVideoFile
        out = node.execute(
            video_path=str(source), settings=settings, codec=codec, container=container,
            quality=quality, filename_prefix=filename_prefix,
            output_directory=output_directory or "", max_frames=0,
            copy_audio=bool(copy_audio), verify_neural_rendering=True,
        )
        values = getattr(out, "result", None) or getattr(out, "args", None) or out
        path, frames = values[0], int(values[1])
        if name_from_source:
            path = _rename_like_source(Path(str(path)), source, filename_prefix)
        return {"ui": {"glide_dlss5_done": [{"path": str(path), "frames": frames}]},
                "result": (str(path), frames)}


# ------------------------------------------------------------- batch node

def _resolve_folder(value: str) -> Path:
    """A folder inside ComfyUI's input, output or temp folder - the same
    confinement as a single video. Relative names live in the output folder,
    where renders are (so "Static" means output/Static)."""
    raw = str(value or "").strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Type the folder that holds your clips, e.g. Static "
                         "(inside ComfyUI's output folder) or its full path.")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = Path(folder_paths.get_output_directory()) / raw
    candidate = candidate.resolve()
    if not any(candidate.is_relative_to(root) for root in _allowed_roots()):
        raise ValueError("Glide DLSS5 Batch only reads folders inside ComfyUI's "
                         "input, output or temp folders.")
    if not candidate.is_dir():
        raise FileNotFoundError(f"No folder at {candidate}")
    return candidate


def _is_our_output(stem: str, tag: str) -> bool:
    """True for a file this node already wrote: <name>_<tag> or <name>_<tag>_N.
    Keeps a batch from upscaling its own results when they share a folder."""
    if not tag:
        return False
    if stem.endswith("_" + tag):
        return True
    head, _, tail = stem.rpartition("_")
    return tail.isdigit() and head.endswith("_" + tag)


class CSGlideDLSS5Batch:
    """Every video in one folder through DLSS5, one after another, each saved
    under its own name with the suffix: 01_1_the-wake_00001.mp4 ->
    01_1_the-wake_00001_DLSS5.mkv. Files are taken in name order."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "settings": ("DLSS5_SETTINGS", {"tooltip": "Connect a DLSS5 Settings node."}),
                "folder": ("STRING", {"default": "",
                           "tooltip": "Folder with your clips. A name like Static means output/Static; "
                                      "a full path works if it is inside ComfyUI's input, output or temp folder."}),
                "codec": (_choices("CODECS", ["HEVC", "H.264", "AV1", "ProRes Proxy"]), {"default": "HEVC"}),
                "container": (_choices("CONTAINERS", ["MKV", "MP4", "MOV"]), {"default": "MKV"}),
                "quality": (_choices("QUALITIES", ["Auto", "Good", "Best", "Max"]), {"default": "Auto"}),
                "suffix": ("STRING", {"default": "DLSS5", "tooltip": "Added to each file name: clip.mp4 -> clip_DLSS5.mkv"}),
                "copy_audio": ("BOOLEAN", {"default": True}),
                "skip_done": ("BOOLEAN", {"default": True,
                              "tooltip": "Skip clips that already have a result with this suffix in the output folder, "
                                         "so a stopped batch picks up where it left off."}),
            },
            "optional": {
                "output_directory": ("STRING", {"default": "",
                                     "tooltip": "Empty = a DLSS5 folder inside the clips folder (e.g. output/Static/DLSS5)."}),
                "name_filter": ("STRING", {"default": "",
                                "tooltip": "Only files whose name contains this text. Empty = every video in the folder."}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("video_paths", "count")
    FUNCTION = "render_all"
    CATEGORY = "CGlide"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # always look at the folder again

    def render_all(self, settings, folder, codec, container, quality, suffix,
                   copy_audio, skip_done, output_directory="", name_filter=""):
        import comfy.model_management as mm
        import comfy.utils

        src_dir = _resolve_folder(folder)
        tag = "".join(c for c in (suffix or "").strip() if c not in '<>:"/\\|?*')
        out_dir = Path(output_directory.strip().strip('"')) if (output_directory or "").strip() \
            else src_dir / "DLSS5"
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = "." + str(container).lower()

        needle = (name_filter or "").strip().lower()
        clips = sorted(
            f for f in src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
            and not _is_our_output(f.stem, tag)
            and (not needle or needle in f.name.lower())
        )
        if not clips:
            raise ValueError(f"No videos to render in {src_dir}.")

        node = _Engine.mod("nodes.enhance_video").DLSS5EnhanceVideoFile
        bar = comfy.utils.ProgressBar(len(clips))
        done, skipped, failed = [], [], []

        for i, clip in enumerate(clips, 1):
            mm.throw_exception_if_processing_interrupted()
            final = out_dir / (clip.stem + (("_" + tag) if tag else "") + ext)
            if skip_done and final.exists():
                print(f"[Glide DLSS5 Batch] {i}/{len(clips)} skip, already done: {final.name}")
                skipped.append(str(final)); bar.update(1)
                continue
            print(f"[Glide DLSS5 Batch] {i}/{len(clips)} {clip.name}")
            try:
                out = node.execute(
                    video_path=str(clip), settings=settings, codec=codec, container=container,
                    quality=quality, filename_prefix="GlideDLSS5batch",
                    output_directory=str(out_dir), max_frames=0,
                    copy_audio=bool(copy_audio), verify_neural_rendering=True,
                )
                values = getattr(out, "result", None) or getattr(out, "args", None) or out
                done.append(_rename_like_source(Path(str(values[0])), clip, tag))
            except mm.InterruptProcessingException:
                raise
            except Exception as error:
                print(f"[Glide DLSS5 Batch] FAILED {clip.name}: {error}")
                failed.append(clip.name)
            bar.update(1)

        print(f"[Glide DLSS5 Batch] finished: {len(done)} rendered, {len(skipped)} skipped, "
              f"{len(failed)} failed -> {out_dir}")
        if failed:
            print("[Glide DLSS5 Batch] failed: " + ", ".join(failed))
        paths = done + skipped
        return {"ui": {"text": [f"{len(done)} rendered, {len(skipped)} skipped, {len(failed)} failed -> {out_dir}"]},
                "result": ("\n".join(paths), len(done))}


class CSGlideDLSS5Legacy(CSGlideDLSS5):
    """The same node under the id the standalone ComfyUI-Glide-DLSS5 folder
    used, so workflows saved with it still open. Hidden from search."""
    DEPRECATED = True


NODE_CLASS_MAPPINGS = {
    "CSGlideDLSS5CS": CSGlideDLSS5,
    "CSGlideDLSS5BatchCS": CSGlideDLSS5Batch,
    "CSGlideDLSS5": CSGlideDLSS5Legacy,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CSGlideDLSS5CS": "Glide DLSS5",
    "CSGlideDLSS5BatchCS": "Glide DLSS5 Batch",
    "CSGlideDLSS5": "Glide DLSS5 (old)",
}
