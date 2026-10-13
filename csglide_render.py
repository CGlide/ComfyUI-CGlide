"""
Server-side Render all for H3 Studio.

Why this exists
---------------
Render all used to be a loop in the browser panel: switch to a clip, take the
previous clip's render into CONTINUE FROM, queue the graph, poll until it is
done, move on. Every step after the first render needed the tab, so closing it
(or letting the laptop sleep) stopped a long project after the clip that was
on the sampler at the time.

The panel now hands the whole chain to the server in one request and only
watches it. Everything the loop did between clips happens here instead, with
the same rules:

  * a LINKED clip takes the tail of its neighbour's render as CONTINUE FROM,
    the last 22 frames on the 17k+5 grid, and the same tail again as the seam
    reference in video 3 (when the node's setting asks for it);
  * an UNLINKED clip can carry the previous render's look into video 1, sound
    off, trimmed to the clearest few frames the span picker finds;
  * each clip is queued through ComfyUI's own /prompt handler, so validation,
    queue order and history are exactly what a click on Queue would get.

Routes:

    POST /cglide/render_job          start a chain, returns {"id": ...}
    GET  /cglide/render_job?id=...   the job as it stands (clip results included)
    POST /cglide/render_job/stop     {"id": ..., "now": false|true}

Jobs live in memory. The chain runs for as long as ComfyUI does: a restart
ends it, and the clips already finished are still in output/ as usual.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import time
import uuid

FPS = 24

# What a continuation guide takes, in frames. The tested 22 -- see tailWindow
# in the panel, which this mirrors.
GUIDE_FRAMES = 22
# What a look carry keeps: the first legal rung. Mirrors LOOK_CARRY_FRAMES.
LOOK_CARRY_FRAMES = 5
# Video slot the seam reference owns (video 3) and the one a look carry owns
# (video 1). Mirrors SEAM_REF_SLOT and the panel's convention.
SEAM_REF_SLOT = 2
LOOK_SLOT = 0

POLL_SECONDS = 1.5
# A prompt that is in neither the queue nor the history this many polls in a
# row was deleted from the queue by hand.
VANISHED_POLLS = 4
KEEP_FINISHED_JOBS = 20

VIDEO_OUT_RE = re.compile(r"\.(mkv|mp4|m4v|mov|webm|avi|mpg|mpeg|ts|m2ts|wmv|flv|ogv)$", re.I)
PREVIEW_RE = re.compile(r"(^|[._-])preview\.[^.]+$", re.I)


# --------------------------------------------------------------------------
# the panel's clip arithmetic, in Python
# --------------------------------------------------------------------------

def grid_spans(dur):
    """Every 17k+5 run the clip can supply, as (frames, seconds)."""
    out = []
    g = 5
    while True:
        sp = (g - 1) / FPS
        if sp > (dur or 0):
            break
        out.append((g, sp))
        g += 17
    return out


def tail_window(dur, want=GUIDE_FRAMES):
    """The last `want` frames on the grid, or the largest run the clip has."""
    dur = float(dur or 0)
    spans = grid_spans(dur)
    pick = next((s for s in spans if s[0] == want), spans[-1] if spans else None)
    if pick is None:
        return 0.0, dur
    return max(0.0, dur - pick[1]), dur


def look_window(span, dur):
    """Keep the span picker's moment, drop its length -- see applySmartSpan."""
    # Math.round rounds halves up; Python's round() would send them to even.
    js_round = lambda x: math.floor(x + 0.5)
    mid = (float(span["start"]) + float(span["end"])) / 2
    first = max(0, js_round(mid * FPS) - LOOK_CARRY_FRAMES // 2)
    last_start = max(0, js_round((dur or 0) * FPS) - LOOK_CARRY_FRAMES)
    start = min(first, last_start) / FPS
    return start, start + (LOOK_CARRY_FRAMES + 0.5) / FPS


def fill_from_adopt(slot, got, silent=False, reference=False):
    """What adoptInto writes into a slot, given the adopt result."""
    slot.clear()
    name = f"{got['subfolder']}/{got['name']}" if got.get("subfolder") else got["name"]
    dur = float(got.get("duration") or 0)
    slot.update({
        "file": name, "dur": dur, "start": 0, "end": dur,
        "audio": not silent,
        "chroma": "" if reference else (got.get("chroma") or ""),
        "codec": "" if reference else (got.get("codec") or ""),
    })
    return slot


def video_from_history(entry):
    """The master video a render wrote. Mirrors videoFromHistory in the panel:
    Glide Video reports the proxy as the file and the real render as `master`."""
    best = proxy = None
    for bag in ((entry or {}).get("outputs") or {}).values():
        for arr in (bag or {}).values():
            if not isinstance(arr, list):
                continue
            for item in arr:
                if not isinstance(item, dict):
                    continue
                master = item.get("master")
                if isinstance(master, str) and VIDEO_OUT_RE.search(master):
                    best = {"name": master, "subfolder": item.get("subfolder") or "",
                            "type": "output"}
                    continue
                fn = item.get("filename")
                if not isinstance(fn, str) or not VIDEO_OUT_RE.search(fn):
                    continue
                hit = {"name": fn, "subfolder": item.get("subfolder") or "",
                       "type": item.get("type") or "output"}
                if PREVIEW_RE.search(fn):
                    proxy = hit
                else:
                    best = hit
    if not best and proxy:
        print("[H3 Studio] Render all: only a preview proxy was found for this "
              "render, chaining from it")
        return proxy
    return best


def _replace_string(obj, old, new):
    """Every occurrence of the exact string `old` in a JSON tree, swapped.

    Used on the workflow copy that rides along as metadata, so the file each
    clip writes carries that clip's h3_data and not whichever clip happened to
    be on screen when the run was started."""
    if isinstance(obj, str):
        return new if obj == old else obj
    if isinstance(obj, list):
        return [_replace_string(x, old, new) for x in obj]
    if isinstance(obj, dict):
        return {k: _replace_string(v, old, new) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# ComfyUI glue, kept behind small functions so the chain can be tested
# against a stand-in
# --------------------------------------------------------------------------

class _Request:
    """Just enough of an aiohttp request for ComfyUI's /prompt handler."""

    def __init__(self, data):
        self._data = data
        self.headers = {}
        self.query = {}

    async def json(self):
        return self._data


class ComfyBackend:
    """Queue, history and file hops, against the running ComfyUI."""

    def __init__(self, server):
        self.server = server

    def _prompt_handler(self):
        for route in self.server.routes:
            if getattr(route, "method", "") == "POST" and getattr(route, "path", "") == "/prompt":
                return route.handler
        return None

    async def submit(self, prompt, extra_data, client_id):
        """Queue one prompt the way the Queue button does. Returns its id.

        Going through ComfyUI's own handler, rather than re-implementing it,
        keeps validation, node replacement and the queue tuple layout whatever
        this ComfyUI version happens to use."""
        handler = self._prompt_handler()
        if handler is None:
            raise RuntimeError("ComfyUI's /prompt route was not found")
        body = {"prompt": prompt, "extra_data": extra_data}
        if client_id:
            body["client_id"] = client_id
        resp = await handler(_Request(body))
        try:
            data = json.loads(resp.body)
        except Exception:
            data = {}
        if resp.status != 200 or not data.get("prompt_id"):
            err = data.get("error") or {}
            msg = err.get("message") if isinstance(err, dict) else str(err)
            details = []
            for nid, ne in (data.get("node_errors") or {}).items():
                for e in (ne or {}).get("errors") or []:
                    details.append(f"node {nid}: {e.get('message')} {e.get('details') or ''}".strip())
            raise RuntimeError("ComfyUI refused the prompt: " + (msg or f"status {resp.status}")
                               + ("\n" + "\n".join(details) if details else ""))
        return data["prompt_id"]

    def history(self, prompt_id):
        return self.server.prompt_queue.get_history(prompt_id=prompt_id).get(prompt_id)

    def _queues(self):
        q = self.server.prompt_queue
        get = getattr(q, "get_current_queue_volatile", None) or q.get_current_queue
        return get()

    def state_of(self, prompt_id):
        """'running', 'pending' or None."""
        running, pending = self._queues()
        if any(len(x) > 1 and x[1] == prompt_id for x in running):
            return "running"
        if any(len(x) > 1 and x[1] == prompt_id for x in pending):
            return "pending"
        return None

    def cancel(self, prompt_id):
        """Take a pending prompt off the queue, or interrupt it if it runs."""
        q = self.server.prompt_queue
        if q.delete_queue_item(lambda a: len(a) > 1 and a[1] == prompt_id):
            return
        stop = getattr(q, "interrupt_if_running", None)
        if stop is not None:
            stop(prompt_id)
        elif self.state_of(prompt_id) == "running":
            import nodes
            nodes.interrupt_processing()

    def adopt(self, pick):
        try:
            from . import csglide_adopt as _adopt
        except ImportError:
            import csglide_adopt as _adopt
        got = _adopt.adopt_file(pick.get("name"), pick.get("subfolder") or "",
                                pick.get("type") or "output")
        if got is None:
            raise RuntimeError(f"the render {pick.get('name')} is no longer in the output folder")
        return got

    def pick_span(self, ref):
        try:
            from . import csglide_cast as _cast
        except ImportError:
            import csglide_cast as _cast
        return _cast.pick_reference_span_cached(ref)


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------

class Stop(Exception):
    pass


class RenderJob:
    def __init__(self, spec, backend):
        self.id = uuid.uuid4().hex
        self.backend = backend
        self.created = time.time()
        self.status = "running"         # running / done / stopped / error
        self.error = ""
        self.note = "starting"
        self.stop = False
        self.force = False
        self.k = 0
        self.prompt_id = None
        self.task = None

        self.prompt = spec["prompt"]
        self.cast_id = str(spec["cast_id"])
        self.extra_data = spec.get("extra_data") or {}
        self.client_id = spec.get("client_id") or None
        self.h3_original = spec.get("h3_original")
        self.project = str(spec.get("project") or "")
        self.count = int(spec.get("count") or 0)
        self.seam_ref = bool(spec.get("seam_ref", True))
        self.look = bool(spec.get("look"))
        self.key = str(spec.get("key") or "")
        self.clips = []
        for c in spec["clips"]:
            self.clips.append({
                "id": str(c.get("id") or ""),
                "index": int(c["index"]),
                "name": str(c.get("name") or ""),
                "link": bool(c.get("link")),
                "state": copy.deepcopy(c.get("state") or {}),
                # the neighbour's render as the project knew it at the start
                "pred": c.get("pred") or None,
                "status": "pending",
                "lastOut": None,
                "prompt_id": None,
            })

    # -- what the panel reads ------------------------------------------------

    def snapshot(self):
        return {
            "id": self.id, "key": self.key, "status": self.status,
            "error": self.error, "note": self.note, "k": self.k,
            "total": len(self.clips), "stopping": self.stop,
            "look": self.look, "created": self.created,
            "clips": [{k: c[k] for k in ("id", "index", "name", "status", "lastOut")}
                      | ({"state": c["state"]} if c["status"] == "done" else {})
                      for c in self.clips],
        }

    def label(self, clip):
        return clip["name"] or f"Clip {clip['index'] + 1}"

    # -- the chain -------------------------------------------------------------

    def _pred_render(self, clip):
        """The render a clip continues from: made earlier in this run if its
        neighbour was in it, otherwise whatever the project had recorded."""
        for c in self.clips:
            if c["index"] == clip["index"] - 1 and c["lastOut"]:
                return c["lastOut"]
        pred = clip.get("pred") or {}
        return pred.get("lastOut") or None

    async def _off_loop(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(None, fn, *args)

    async def prepare(self, clip):
        """Fill the slots the way the panel's loop did, before queueing."""
        st = clip["state"]
        slots = st.setdefault("slots", {})
        videos = slots.setdefault("videos", [])
        while len(videos) < 3:
            videos.append({})
        cont = st.setdefault("cont", {})

        has_pred = clip["index"] > 0
        linked = has_pred and clip["link"]
        src = self._pred_render(clip) if has_pred else None
        pred_name = (clip.get("pred") or {}).get("name") or f"Clip {clip['index']}"

        if linked and src:
            self.note = f"taking the tail of {pred_name}"
            got = await self._off_loop(self.backend.adopt, src)
            fill_from_adopt(cont, got)
            start, end = tail_window(cont["dur"], GUIDE_FRAMES)
            cont["start"], cont["end"] = start, end
            if self.seam_ref:
                self.note = "doubling the tail as a seam reference"
                slot = videos[SEAM_REF_SLOT]
                if slot.get("file") and not slot.get("seam"):
                    print(f"[H3 Studio] seam reference: replacing {slot['file']} in video "
                          f"{SEAM_REF_SLOT + 1} - the continuation owns that slot")
                # The guide's own copy, as when a clip is dropped on CONTINUE
                # FROM: the same file, so the window transfers directly.
                slot.clear()
                slot.update({"file": cont["file"], "dur": cont["dur"],
                             "chroma": cont.get("chroma", ""), "codec": cont.get("codec", ""),
                             "audio": False, "start": start, "end": end, "seam": True})
        elif linked:
            if not cont.get("file"):
                raise RuntimeError(f"{self.label(clip)} is linked but {pred_name} has no "
                                   "render to continue from")
            print(f"[H3 Studio] Render: {self.label(clip)} is linked but no render of "
                  f"{pred_name} is recorded - keeping the CONTINUE FROM already set "
                  f"({cont['file']})")

        # The look is for cuts only; see renderAll in the panel.
        if not linked and self.look and has_pred:
            if not src:
                print(f"[H3 Studio] Render: no render of {pred_name} recorded, no look "
                      f"to carry into {self.label(clip)}")
            else:
                self.note = f"carrying the look from {pred_name}"
                slot = videos[LOOK_SLOT]
                got = await self._off_loop(self.backend.adopt, src)
                fill_from_adopt(slot, got, silent=True, reference=True)
                slot["carry"] = True
                self.note = "choosing the clearest window"
                span = None
                try:
                    span = await self._off_loop(self.backend.pick_span, slot["file"])
                except Exception as exc:
                    print(f"[H3 Studio] smart span pick failed: {exc}")
                if span:
                    slot["start"], slot["end"] = look_window(span, slot["dur"])
                    slot["spanNote"] = span.get("note") or ""
                    slot["spanScore"] = float(span.get("score") or 0)
                else:
                    slot["spanNote"] = "weak"

    def _prompt_for(self, clip):
        meta = {"name": clip["name"].strip(), "index": clip["index"] + 1,
                "count": self.count, "project": self.project.strip()}
        h3 = json.dumps(dict(clip["state"], clip=meta))
        prompt = copy.deepcopy(self.prompt)
        prompt[self.cast_id]["inputs"]["h3_data"] = h3
        extra = copy.deepcopy(self.extra_data)
        if self.h3_original:
            extra = _replace_string(extra, self.h3_original, h3)
        return prompt, extra

    async def wait(self, prompt_id):
        gone = 0
        while True:
            if self.force:
                raise Stop()
            entry = self.backend.history(prompt_id)
            if entry:
                st = entry.get("status") or {}
                if st.get("status_str") == "error":
                    if self.force or self.stop:
                        raise Stop()
                    msg = ""
                    for m in st.get("messages") or []:
                        if isinstance(m, (list, tuple)) and len(m) > 1 and m[0] == "execution_error":
                            d = m[1] or {}
                            msg = f"{d.get('node_type', '')}: {d.get('exception_message', '')}".strip(": ")
                    raise RuntimeError("the render failed" + (f" - {msg}" if msg else " - see the console"))
                if st.get("completed") or st.get("status_str") == "success":
                    return entry
            elif self.backend.state_of(prompt_id) is None:
                gone += 1
                if gone >= VANISHED_POLLS:
                    raise RuntimeError("the clip was removed from the queue")
            else:
                gone = 0
            await asyncio.sleep(POLL_SECONDS)

    async def run(self):
        try:
            for k, clip in enumerate(self.clips):
                self.k = k
                if self.stop:
                    raise Stop()
                clip["status"] = "preparing"
                await self.prepare(clip)
                if self.force:
                    raise Stop()
                prompt, extra = self._prompt_for(clip)
                self.note = "queued, waiting for the render"
                pid = await self.backend.submit(prompt, extra, self.client_id)
                clip["prompt_id"] = self.prompt_id = pid
                clip["status"] = "rendering"
                entry = await self.wait(pid)
                out = video_from_history(entry)
                if not out:
                    raise RuntimeError("that render produced no video file - is Glide Video "
                                       "(or another video output) connected in the graph?")
                clip["lastOut"] = {"name": out["name"], "subfolder": out.get("subfolder") or "",
                                   "type": out.get("type") or "output"}
                clip["status"] = "done"
                self.prompt_id = None
                print(f"[H3 Studio] Render: {self.label(clip)} ({k + 1}/{len(self.clips)}) "
                      f"-> {out['name']}")
            self.status = "done"
            self.note = ""
            print(f"[H3 Studio] Render: done, {len(self.clips)} clip(s)")
        except Stop:
            self.status = "stopped"
            self.note = ""
            print("[H3 Studio] Render: stopped")
        except Exception as exc:
            self.status = "error"
            self.error = str(exc) or exc.__class__.__name__
            print(f"[H3 Studio] Render stopped at clip {self.k + 1}: {self.error}")
        finally:
            for c in self.clips:
                if c["status"] in ("preparing", "rendering"):
                    c["status"] = "failed" if self.status == "error" else "stopped"
            self.prompt_id = None

    def request_stop(self, now):
        self.stop = True
        if now:
            self.force = True
            if self.prompt_id:
                try:
                    self.backend.cancel(self.prompt_id)
                except Exception as exc:
                    print(f"[H3 Studio] could not cancel {self.prompt_id}: {exc}")


JOBS: dict = {}


def _prune():
    done = [j for j in JOBS.values() if j.status != "running"]
    done.sort(key=lambda j: j.created)
    for j in done[:-KEEP_FINISHED_JOBS] if len(done) > KEEP_FINISHED_JOBS else []:
        JOBS.pop(j.id, None)


def validate_spec(spec):
    if not isinstance(spec, dict):
        return "body is not an object"
    prompt = spec.get("prompt")
    if not isinstance(prompt, dict) or not prompt:
        return "prompt is required"
    cast = prompt.get(str(spec.get("cast_id")))
    if not isinstance(cast, dict) or not isinstance(cast.get("inputs"), dict):
        return "cast_id does not name a node in the prompt"
    clips = spec.get("clips")
    if not isinstance(clips, list) or not clips:
        return "clips is required"
    for c in clips:
        if not isinstance(c, dict) or not isinstance(c.get("index"), int):
            return "every clip needs an index"
    return None


def start_job(spec, backend):
    err = validate_spec(spec)
    if err:
        raise ValueError(err)
    key = str(spec.get("key") or "")
    if key and any(j.key == key and j.status == "running" for j in JOBS.values()):
        raise ValueError("this H3 Studio node is already rendering")
    job = RenderJob(spec, backend)
    JOBS[job.id] = job
    _prune()
    job.task = asyncio.get_running_loop().create_task(job.run())
    return job


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------

try:
    from aiohttp import web
    from server import PromptServer

    _backend = ComfyBackend(PromptServer.instance)

    @PromptServer.instance.routes.post("/cglide/render_job")
    async def _start(request):
        try:
            spec = await request.json()
        except Exception:
            return web.json_response({"error": "body is not JSON"}, status=400)
        try:
            job = start_job(spec, _backend)
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        return web.json_response(job.snapshot())

    @PromptServer.instance.routes.get("/cglide/render_job")
    async def _get(request):
        job = JOBS.get(request.rel_url.query.get("id") or "")
        if job is None:
            return web.json_response({"error": "no such job"}, status=404)
        return web.json_response(job.snapshot())

    @PromptServer.instance.routes.post("/cglide/render_job/stop")
    async def _stop(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        job = JOBS.get(str(data.get("id") or ""))
        if job is None:
            return web.json_response({"error": "no such job"}, status=404)
        if job.status == "running":
            job.request_stop(bool(data.get("now")))
        return web.json_response(job.snapshot())
except Exception as _exc:  # pragma: no cover - outside ComfyUI
    if __name__ != "__main__":
        print("[H3 Studio] /cglide/render_job not registered (%s)" % _exc)
