"""
Server-side routes for taking a finished render back into the input folder.

Why this exists
---------------
A continuation needs the previous clip sitting in ``input/cglide/`` so the node
can read it as a guide. Renders land in ``output/``. Doing that hop in the
browser means downloading a 12-second 4:4:4 10-bit file and uploading the same
bytes straight back through ``/upload/image`` -- hundreds of megabytes across
the loopback for a copy the server could do on disk in a few milliseconds.

Two routes:

    GET  /cglide/recent_outputs?limit=8
        The newest video files under the output folder, newest first.

    POST /cglide/adopt_output
        {"filename": "...", "subfolder": "...", "type": "output"}
        Copies that file into input/<ASSET_SUBFOLDER>/ and reports the name it
        landed under, plus its duration when ffprobe can be found.

Both are read-only with respect to the output folder; nothing is moved or
deleted, so a render stays exactly where the user expects to find it.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess

from aiohttp import web

import folder_paths
from server import PromptServer

# Same folder the node's own uploads go to. Lowercase deliberately: ComfyUI on
# Linux is case-sensitive and a stray capital splits the folder in two.
ASSET_SUBFOLDER = "cglide"

# Containers Glide Video can emit, plus the usual suspects a user might have
# rendered with some other node. Kept as a set of lowercase extensions.
VIDEO_EXT = {
    ".mkv", ".mp4", ".m4v", ".mov", ".webm", ".avi", ".mpg", ".mpeg",
    ".ts", ".mts", ".m2ts", ".wmv", ".flv", ".ogv",
}

# Glide Video writes a browser-playable proxy beside any master a <video> tag
# cannot decode -- a 4:4:4 mkv render arrives with an h264 4:2:0 mp4 twin. The
# proxy is written second, so it is always "newest", and continuing from it
# would hand the next clip a subsampled guide. Masters only unless asked.
_PREVIEW_RE = re.compile(r"(^|[._-])preview\.[^.]+$", re.IGNORECASE)

# A deep output tree with a long history in it should not turn a button press
# into a filesystem crawl. Newest-first ordering only needs mtimes, but we still
# stop after this many candidates.
_SCAN_LIMIT = 4000


# --------------------------------------------------------------------------
# paths
# --------------------------------------------------------------------------

def _root_for(kind: str) -> str:
    """Absolute path of ComfyUI's output / input / temp directory."""
    kind = (kind or "output").lower()
    if kind == "input":
        return folder_paths.get_input_directory()
    if kind == "temp":
        return folder_paths.get_temp_directory()
    return folder_paths.get_output_directory()


def _resolve_inside(root: str, *parts: str) -> str | None:
    """Join under ``root`` and refuse anything that escapes it.

    ``subfolder`` and ``filename`` arrive from the browser, so a request could
    ask for ``../../../etc/passwd``. Resolving symlinks and comparing prefixes
    is the only check that survives ``..`` mixed with links.
    """
    candidate = os.path.realpath(os.path.join(root, *parts))
    root = os.path.realpath(root)
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    return candidate


def _unique_path(directory: str, name: str) -> str:
    """A free path in ``directory``, suffixing ``_1``, ``_2`` ... on collision.

    Adopting the same render twice is normal -- render, look at it, render
    again -- and silently overwriting the earlier copy would pull the rug out
    from under any clip still pointing at it.
    """
    stem, ext = os.path.splitext(name)
    path = os.path.join(directory, name)
    n = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}_{n}{ext}")
        n += 1
    return path


# --------------------------------------------------------------------------
# duration
# --------------------------------------------------------------------------

def _ffprobe_exe() -> str | None:
    """Best effort at an ffprobe binary.

    imageio-ffmpeg ships ffmpeg but NOT ffprobe, so the sibling-of-ffmpeg guess
    usually misses and PATH is what actually answers. Returning None is fine --
    the browser probes the duration itself and this is only the fallback for
    containers it cannot decode.
    """
    found = shutil.which("ffprobe")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: WPS433 - optional dependency
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cand = os.path.join(os.path.dirname(exe),
                            "ffprobe.exe" if os.name == "nt" else "ffprobe")
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    return None


def _chroma(pix_fmt: str) -> str:
    """Chroma subsampling as a label, from ffmpeg's pixel format name.

    Matters because a 4:2:0 guide measurably weakens the continuation anchor:
    tested on the same pair, an AV1 yuv420p10le source made Glide Join report a
    -2 frame correspondence every time, while 4:2:2 and 4:4:4 sources reported
    none. Three quarters of the chroma is gone before the model ever sees the
    frames, so it has less to lock onto.
    """
    p = (pix_fmt or "").lower()
    if p.startswith("gbr") or p.startswith("rgb") or p.startswith("bgr"):
        return "RGB"
    if "444" in p:
        return "4:4:4"
    if "422" in p:
        return "4:2:2"
    if "440" in p:
        return "4:4:0"
    if "420" in p:
        return "4:2:0"
    if "411" in p:
        return "4:1:1"
    return ""


# ffprobe on a hover is fine once; on every hover it is not. Keyed on identity
# AND mtime/size so a rewritten file at the same path is probed again.
_PROBE_CACHE: dict = {}
_PROBE_CACHE_MAX = 256


def _ffmpeg_exe() -> str | None:
    """ffmpeg, which is present far more often than ffprobe.

    imageio-ffmpeg bundles ffmpeg and NOT ffprobe, so on a stock ComfyUI the
    probe binary is usually missing while ffmpeg is right there. Parsing
    `ffmpeg -i` stderr is the fallback the video node already relies on.
    """
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # noqa: WPS433 - optional dependency
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    return None


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #\d+:\d+.*?:\s*Video:\s*([A-Za-z0-9_]+)[^,]*,\s*([A-Za-z0-9]+)")


def _probe_via_ffmpeg(path: str, out: dict) -> None:
    """Fill blanks in `out` from `ffmpeg -i` stderr.

    ffmpeg exits non-zero here because no output file was given -- that is
    expected, and the stream description it prints on the way out is the whole
    point.
    """
    exe = _ffmpeg_exe()
    if not exe:
        return
    try:
        err = subprocess.run([exe, "-hide_banner", "-i", path],
                             capture_output=True, text=True, timeout=20).stderr
    except Exception:
        return
    if not out["duration"]:
        m = _DUR_RE.search(err)
        if m:
            out["duration"] = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    if not out["codec"] or not out["pix_fmt"]:
        m = _VID_RE.search(err)
        if m:
            out["codec"] = out["codec"] or m.group(1)
            out["pix_fmt"] = out["pix_fmt"] or m.group(2)


def _probe(path: str) -> dict:
    """Duration, codec and pixel format. Zeros and blanks when unavailable.

    The duration is worth having on its own: the browser reports 0 for anything
    it cannot decode, and HEVC 4:4:4 in Matroska -- what the high-fidelity
    preset writes -- is exactly that case. Without a duration the trim window
    has nothing to draw.
    """
    try:
        stat = os.stat(path)
        key = (os.path.realpath(path), stat.st_mtime, stat.st_size)
    except OSError:
        key = (path, 0, 0)
    hit = _PROBE_CACHE.get(key)
    if hit is not None:
        return hit

    out = {"duration": 0.0, "codec": "", "pix_fmt": "", "chroma": ""}
    exe = _ffprobe_exe()
    if exe:
        try:
            res = subprocess.run(
                [exe, "-v", "error",
                 "-select_streams", "v:0",
                 "-show_entries", "format=duration:stream=codec_name,pix_fmt",
                 "-of", "default=nw=1", path],
                capture_output=True, text=True, timeout=15,
            ).stdout
            for line in res.splitlines():
                key_, _, val = line.partition("=")
                val = val.strip()
                if key_ == "duration":
                    try:
                        out["duration"] = max(0.0, float(val))
                    except ValueError:
                        pass
                elif key_ == "codec_name":
                    out["codec"] = val
                elif key_ == "pix_fmt":
                    out["pix_fmt"] = val
            out["chroma"] = _chroma(out["pix_fmt"])
        except Exception:
            pass

    # ffprobe missing, or present but silent on some field
    if not out["pix_fmt"] or not out["duration"]:
        _probe_via_ffmpeg(path, out)
        out["chroma"] = _chroma(out["pix_fmt"])

    if len(_PROBE_CACHE) >= _PROBE_CACHE_MAX:
        _PROBE_CACHE.clear()
    _PROBE_CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

@PromptServer.instance.routes.get("/cglide/recent_outputs")
async def _recent_outputs(request: web.Request) -> web.Response:
    """Newest video files in the output folder, newest first."""
    try:
        limit = max(1, min(50, int(request.query.get("limit", "8"))))
    except ValueError:
        limit = 8
    root = _root_for(request.query.get("type", "output"))
    keep_previews = request.query.get("previews") in ("1", "true", "yes")

    def scan():
        found = []
        seen = 0
        for base, dirs, files in os.walk(root):
            dirs.sort()
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in VIDEO_EXT:
                    continue
                if not keep_previews and _PREVIEW_RE.search(fn):
                    continue
                full = os.path.join(base, fn)
                try:
                    stat = os.stat(full)
                except OSError:
                    continue
                rel = os.path.relpath(base, root)
                found.append({
                    "name": fn,
                    "subfolder": "" if rel == "." else rel.replace(os.sep, "/"),
                    "mtime": stat.st_mtime,
                    "size": stat.st_size,
                })
                seen += 1
                if seen >= _SCAN_LIMIT:
                    return found
        return found

    try:
        found = await asyncio.get_running_loop().run_in_executor(None, scan)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    found.sort(key=lambda item: item["mtime"], reverse=True)
    found = found[:limit]

    # Opt-in: the hover tooltip wants codec and chroma BEFORE the file is
    # adopted, so it can say what you are about to pull in. Probing is cached on
    # path+mtime+size, so repeated hovers cost nothing after the first.
    if request.query.get("probe") in ("1", "true", "yes"):
        root_dir = root

        def probe_all():
            for item in found:
                path = _resolve_inside(root_dir, item["subfolder"], item["name"])
                if path:
                    item.update(_probe(path))
            return found

        try:
            found = await asyncio.get_running_loop().run_in_executor(None, probe_all)
        except Exception:
            pass          # the listing is still useful without the details

    return web.json_response(found)


def _find_by_name(root: str, name: str) -> str | None:
    """Newest file with this basename anywhere under ``root``.

    Needed because a master file's name can arrive without its folder. Glide
    Video reports the browser proxy -- a temp file with no subfolder -- and
    names the real render beside it in a ``master`` field, but a
    ``filename_prefix`` like ``video/MiniMax_H3`` puts that master in
    ``output/video/`` while the entry says ``""``. The name is unique enough to
    find; the reported folder is simply not its folder.
    """
    base = os.path.basename(name)
    best = None
    best_mtime = -1.0
    seen = 0
    for dirpath, dirs, files in os.walk(root):
        dirs.sort()
        if base in files:
            full = os.path.join(dirpath, base)
            try:
                mt = os.stat(full).st_mtime
            except OSError:
                continue
            if mt > best_mtime:
                best, best_mtime = full, mt
        seen += len(files)
        if seen >= _SCAN_LIMIT:
            break
    return best


@PromptServer.instance.routes.get("/cglide/probe")
async def _probe_file(request: web.Request) -> web.Response:
    """Codec, pixel format and duration for a file already on disk.

    The adopt route probes what it copies, but a clip dragged onto the slot
    goes through ComfyUI's own upload endpoint and never passes through here --
    which is exactly the case where the chroma is unknown, since it is someone
    else's file rather than one of our own presets.
    """
    filename = str(request.query.get("filename") or "")
    if not filename:
        return web.json_response({"error": "filename is required"}, status=400)
    root = _root_for(request.query.get("type", "input"))
    path = _resolve_inside(root, request.query.get("subfolder", ""), filename)
    if not path or not os.path.isfile(path):
        return web.json_response({"error": "file not found"}, status=404)
    try:
        info = await asyncio.get_running_loop().run_in_executor(None, _probe, path)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)
    return web.json_response(info)


@PromptServer.instance.routes.post("/cglide/adopt_output")
async def _adopt_output(request: web.Request) -> web.Response:
    """Copy one output file into input/<ASSET_SUBFOLDER>/ and report its name."""
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return web.json_response({"error": "body is not JSON"}, status=400)

    filename = str(data.get("filename") or "")
    subfolder = str(data.get("subfolder") or "")
    if not filename:
        return web.json_response({"error": "filename is required"}, status=400)

    root = _root_for(data.get("type", "output"))
    src = _resolve_inside(root, subfolder, filename)
    if not src or not os.path.isfile(src):
        # Not where we were told. Look for it by name before giving up -- see
        # _find_by_name for why the reported folder can be wrong.
        found = await asyncio.get_running_loop().run_in_executor(
            None, _find_by_name, root, filename)
        if found:
            src = found
        else:
            # Same answer for "outside the root" and "not there": a probe should
            # not learn which paths exist from the status code.
            return web.json_response({"error": "file not found"}, status=404)

    dest_dir = os.path.join(folder_paths.get_input_directory(), ASSET_SUBFOLDER)
    try:
        os.makedirs(dest_dir, exist_ok=True)
        dest = _unique_path(dest_dir, os.path.basename(src))
        loop = asyncio.get_running_loop()
        # copy2, not copy: a 12s 4:4:4 render is big enough that doing it on the
        # event loop would stall every other request for the duration.
        await loop.run_in_executor(None, shutil.copy2, src, dest)
        info = await loop.run_in_executor(None, _probe, dest)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)

    print(f"[H3 Studio] adopted {os.path.basename(src)} "
          f"-> input/{ASSET_SUBFOLDER}/{os.path.basename(dest)}"
          + (f"  ({info['codec']} {info['chroma']})" if info["codec"] else ""))
    return web.json_response({
        "name": os.path.basename(dest),
        "subfolder": ASSET_SUBFOLDER,
        "duration": info["duration"],
        "codec": info["codec"],
        "pix_fmt": info["pix_fmt"],
        "chroma": info["chroma"],
    })
