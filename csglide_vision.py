import os
import io
import re
import json
import base64
import asyncio
import socket
import ipaddress
import urllib.request
import urllib.error
from uuid import uuid4
from urllib.parse import urlsplit

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths
from server import PromptServer
from aiohttp import web


# ----------------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------------
DEFAULT_PROMPT = (
    "Describe this image in 3 parts. "
    "the main subject." "the setting/background. "
    "the style, lighting or mood. Be specific and concise."
)
DEFAULT_URL = "http://localhost:11434"
DEFAULT_MODEL = "llava"


# ----------------------------------------------------------------------------
# SSRF guard
#
# Every route below takes a URL from whoever called it and makes THIS machine
# fetch it. Without a guard that is server-side request forgery: anyone who can
# reach your ComfyUI can use it as a proxy into your own network. On a rented
# GPU box http://169.254.169.254/ is the cloud metadata service and hands out
# credentials; on a LAN it is a port scanner; and urllib speaks file:// as well
# as http, so an unchecked URL reads local disk too.
#
# Two different policies, because the two features want different things:
#
#   BACKEND (analyze / models / eject) -- talks to a vision server. Loopback is
#   the normal case (Ollama, LM Studio) and a public cloud endpoint is a valid
#   case (any OpenAI-compatible API). What it must NOT reach is the rest of the
#   private network or the metadata service.
#
#   IMAGE FETCH (load_url) -- pulls a picture off the public web. It has no
#   business touching loopback or anything private at all.
#
# Running Ollama on another machine on your LAN is a real setup and it is
# blocked by default, so there is an explicit opt-in:
#
#   set CSGLIDE_VISION_ALLOW_HOSTS=192.168.1.50,ollama.lan
#
# Hosts listed there skip the private-address check. It is deliberately an
# environment variable and not a node widget: a value that arrives in the
# request is exactly what an attacker controls, so it could not be a guard.
#
# Known limits, stated rather than hidden: the hostname is resolved and checked
# before the request, so a DNS name that resolves differently a moment later
# (DNS rebinding) is not covered, and redirects are re-checked per hop but the
# final connection still goes through urllib's own resolution. Closing that
# properly needs connecting to a pinned IP, which breaks TLS hostname
# verification and https proxies. What is here stops the whole class of
# one-shot attacks against a fixed internal address.
# ----------------------------------------------------------------------------
ALLOWED_SCHEMES = ("http", "https")
MAX_FETCH_BYTES = 32 * 1024 * 1024          # a picture, not a disk image


class UrlNotAllowed(RuntimeError):
    """The URL is refused by the guard. Message is safe to show the user."""


def _extra_allowed_hosts():
    raw = os.environ.get("CSGLIDE_VISION_ALLOW_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _is_internal(ip):
    """True for anything that is not a normal public address."""
    if ip.version == 6:
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
    )


def _resolve(host, port):
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UrlNotAllowed("Can't resolve '%s'." % host)
    out = []
    for info in infos:
        try:
            out.append(ipaddress.ip_address(info[4][0]))
        except ValueError:
            continue
    if not out:
        raise UrlNotAllowed("Can't resolve '%s'." % host)
    return out


def check_url(url, allow_loopback):
    """Raise UrlNotAllowed unless this URL is safe to fetch. Returns the url."""
    parts = urlsplit(url or "")
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UrlNotAllowed(
            "Only http and https URLs are allowed (got '%s')." % (scheme or "none")
        )
    host = (parts.hostname or "").lower()
    if not host:
        raise UrlNotAllowed("That URL has no host in it.")

    if host in _extra_allowed_hosts():
        return url                      # explicit opt-in by whoever runs the server

    port = parts.port or (443 if scheme == "https" else 80)
    for ip in _resolve(host, port):
        if ip.is_loopback:
            if allow_loopback:
                continue
            raise UrlNotAllowed(
                "This one only fetches from the public web, not from this machine."
            )
        if _is_internal(ip):
            raise UrlNotAllowed(
                "'%s' is a private or internal address, so it is blocked. If it is "
                "your own vision server, add its host to CSGLIDE_VISION_ALLOW_HOSTS "
                "and restart ComfyUI." % host
            )
    return url


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """A public URL that 302s to 169.254.169.254 would walk straight past a
    check done only on the URL you typed, so every hop gets checked too."""

    def __init__(self, allow_loopback):
        self.allow_loopback = allow_loopback

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.allow_loopback)
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )


def _opener(allow_loopback):
    return urllib.request.build_opener(_GuardedRedirects(allow_loopback))


def guarded_open(req, timeout, allow_loopback):
    """urlopen, with the URL and every redirect checked first."""
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    check_url(url, allow_loopback)
    return _opener(allow_loopback).open(req, timeout=timeout)


def _read_capped(resp):
    data = resp.read(MAX_FETCH_BYTES + 1)
    if len(data) > MAX_FETCH_BYTES:
        raise UrlNotAllowed(
            "That file is over %d MB, which is too big to load."
            % (MAX_FETCH_BYTES // (1024 * 1024))
        )
    return data


def _is_loopback_url(url):
    """Whether error text from this host is safe to hand back verbatim."""
    try:
        host = (urlsplit(url or "").hostname or "").lower()
        if not host:
            return False
        return all(ip.is_loopback for ip in _resolve(host, 80))
    except Exception:
        return False


# ----------------------------------------------------------------------------
# Image helpers
# ----------------------------------------------------------------------------
def pil_to_tensor(img):
    img = img.convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]  # (1, H, W, 3)


def load_image_from_input(name):
    name = (name or "").strip()
    if not name:
        return None
    input_dir = folder_paths.get_input_directory()
    path = os.path.join(input_dir, name)
    if not os.path.exists(path):
        try:
            path = folder_paths.get_annotated_filepath(name)
        except Exception:
            return None
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    return img


# ----------------------------------------------------------------------------
# Backends (run in a thread executor so the UI never freezes)
# ----------------------------------------------------------------------------
def _call_ollama(url, model, prompt, image_b64):
    endpoint = url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with guarded_open(req, 600, allow_loopback=True) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as he:
        # Only quote the response body back when it came from this machine.
        # From anywhere else that body is someone else's content and echoing it
        # turns a blocked request into a readable one.
        body = ""
        if _is_loopback_url(endpoint):
            try:
                body = he.read().decode("utf-8")[:2000]
            except Exception:
                pass
        if he.code == 404:
            raise RuntimeError(
                "Ollama can't find model '%s'. Pull it with `ollama pull %s`, "
                "or pick an installed model in the gear panel." % (model, model)
            )
        raise RuntimeError("Ollama HTTP %s: %s" % (he.code, body or "request failed"))
    except urllib.error.URLError as ue:
        raise RuntimeError(
            "Can't reach Ollama at %s (%s). Is Ollama running?" % (url, ue.reason)
        )
    return (out.get("response") or "").strip()


def _call_openai(url, model, prompt, image_b64, api_key=""):
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "max_tokens": 1024,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers=headers)
    with guarded_open(req, 600, allow_loopback=True) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return (out["choices"][0]["message"]["content"] or "").strip()


def run_analysis(image_b64, prompt, backend, url, model, api_key=""):
    prompt = (prompt or "").strip() or DEFAULT_PROMPT
    backend = (backend or "ollama").lower()
    url = (url or DEFAULT_URL).strip()
    model = (model or DEFAULT_MODEL).strip()
    if backend == "openai":
        return _call_openai(url, model, prompt, image_b64, api_key)
    return _call_ollama(url, model, prompt, image_b64)


# ----------------------------------------------------------------------------
# API routes used by the Analyze button / URL loader in the browser
# ----------------------------------------------------------------------------
routes = PromptServer.instance.routes


def _err(e, where):
    """One error shape for the routes. UrlNotAllowed is written to be read by
    the user; anything else gets logged here and summarised there, so a stack
    trace or a fetched response body never travels back over the wire."""
    if isinstance(e, (UrlNotAllowed, RuntimeError)):
        return str(e)
    print("[Glide Vision] %s failed: %r" % (where, e))
    return "That didn't work - see the ComfyUI console for the details."


@routes.post("/csglide_vision/analyze")
async def _analyze(request):
    try:
        data = await request.json()
        image_b64 = data.get("image", "")
        if not image_b64:
            name = data.get("name", "")
            pil = load_image_from_input(name)
            if pil is None:
                return web.json_response(
                    {"ok": False, "error": "No image provided."}, status=200
                )
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="PNG")
            image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None,
            run_analysis,
            image_b64,
            data.get("prompt", ""),
            data.get("backend", "ollama"),
            data.get("url", DEFAULT_URL),
            data.get("model", DEFAULT_MODEL),
            data.get("api_key", ""),
        )
        return web.json_response({"ok": True, "description": text})
    except Exception as e:
        return web.json_response({"ok": False, "error": _err(e, "analyze")}, status=200)


_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,text/html;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def _origin(u):
    p = urlsplit(u)
    if p.scheme and p.netloc:
        return "%s://%s/" % (p.scheme, p.netloc)
    return None


def _fetch_url_bytes(url, referer=None):
    headers = dict(_BROWSER_HEADERS)
    headers["Referer"] = referer or _origin(url) or ""
    req = urllib.request.Request(url, headers=headers)
    # allow_loopback=False: this is the public-web fetcher. Nothing about
    # loading a reference picture needs to reach this machine.
    with guarded_open(req, 60, allow_loopback=False) as r:
        ctype = r.headers.get("Content-Type", "") or ""
        return _read_capped(r), ctype


def _extract_og_image(html):
    # Pinterest pin pages (and most sites) expose the real image via og:image
    patterns = [
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
    ]
    for p in patterns:
        m = re.search(p, html, re.I)
        if m:
            return m.group(1).replace("&amp;", "&")
    return None


@routes.post("/csglide_vision/load_url")
async def _load_url(request):
    try:
        data = await request.json()
        url = (data.get("url") or "").strip()
        if not url:
            return web.json_response({"ok": False, "error": "Empty URL."}, status=200)

        # Fail before fetching, so a refused URL says why instead of timing out.
        check_url(url, allow_loopback=False)

        loop = asyncio.get_event_loop()
        ref = "https://www.pinterest.com/" if "pin" in url.lower() else None

        raw, ctype = await loop.run_in_executor(None, _fetch_url_bytes, url, ref)

        img = None
        if "html" not in ctype.lower():
            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                img = None

        # got a web page instead of an image -> dig out the real image URL.
        # The og:image URL comes off a fetched page, so it is no more trusted
        # than the one that was typed in; guarded_open checks it as well.
        if img is None:
            og = None
            try:
                og = _extract_og_image(raw.decode("utf-8", "ignore"))
            except Exception:
                og = None
            if og:
                raw2, _ = await loop.run_in_executor(None, _fetch_url_bytes, og, ref)
                img = Image.open(io.BytesIO(raw2)).convert("RGB")

        if img is None:
            return web.json_response(
                {
                    "ok": False,
                    "error": (
                        "Couldn't read an image from that link. On Pinterest, "
                        "right-click the image and choose 'Copy image address' "
                        "(an i.pinimg.com link), then paste that."
                    ),
                },
                status=200,
            )

        fname = f"csglide_{uuid4().hex[:8]}.png"
        img.save(os.path.join(folder_paths.get_input_directory(), fname))

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return web.json_response({"ok": True, "name": fname, "image_b64": b64})
    except Exception as e:
        return web.json_response({"ok": False, "error": _err(e, "load_url")}, status=200)


@routes.get("/csglide_vision/models")
async def _models(request):
    url = request.query.get("url", DEFAULT_URL)

    def _tags():
        endpoint = url.rstrip("/") + "/api/tags"
        req = urllib.request.Request(endpoint)
        with guarded_open(req, 15, allow_loopback=True) as r:
            return json.loads(_read_capped(r).decode("utf-8"))

    try:
        check_url(url, allow_loopback=True)
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(None, _tags)
        names = [m.get("name", "") for m in data.get("models", [])]
        return web.json_response({"ok": True, "models": [n for n in names if n]})
    except Exception as e:
        return web.json_response(
            {"ok": False, "error": _err(e, "models"), "models": []}, status=200
        )


@routes.post("/csglide_vision/eject")
async def _eject(request):
    try:
        data = await request.json()
        backend = (data.get("backend") or "ollama").lower()
        if backend != "ollama":
            return web.json_response(
                {"ok": True, "msg": "Eject only applies to Ollama."}
            )
        url = (data.get("url") or DEFAULT_URL).strip()
        model = (data.get("model") or DEFAULT_MODEL).strip()
        check_url(url, allow_loopback=True)

        def _unload():
            endpoint = url.rstrip("/") + "/api/generate"
            payload = {"model": model, "keep_alive": 0}
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint, data=body, headers={"Content-Type": "application/json"}
            )
            with guarded_open(req, 60, allow_loopback=True) as r:
                return _read_capped(r)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _unload)
        return web.json_response({"ok": True, "msg": "Ejected '%s'." % model})
    except Exception as e:
        return web.json_response({"ok": False, "error": _err(e, "eject")}, status=200)


# ----------------------------------------------------------------------------
# The node (graph-side): turns the loaded image + description into outputs
# ----------------------------------------------------------------------------
class CSGlideVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_name": ("STRING", {"default": ""}),
                "custom_prompt": ("STRING", {"default": "", "multiline": True}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "backend": (["ollama", "openai"], {"default": "ollama"}),
                "server_url": ("STRING", {"default": DEFAULT_URL}),
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "auto_eject": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "description")
    FUNCTION = "execute"
    CATEGORY = "CSGlide"

    def execute(self, image_name, custom_prompt, description, backend, server_url, model, auto_eject=False):
        if auto_eject and (backend or "").lower() == "ollama":
            try:
                endpoint = (server_url or DEFAULT_URL).rstrip("/") + "/api/generate"
                body = json.dumps(
                    {"model": (model or DEFAULT_MODEL), "keep_alive": 0}
                ).encode("utf-8")
                req = urllib.request.Request(
                    endpoint, data=body, headers={"Content-Type": "application/json"}
                )
                with guarded_open(req, 60, allow_loopback=True) as r:
                    r.read(1024)
            except UrlNotAllowed as e:
                print("[Glide Vision] auto_eject skipped: %s" % e)
            except Exception:
                pass

        img = load_image_from_input(image_name)
        if img is not None:
            out_image = pil_to_tensor(img)
        else:
            out_image = torch.zeros((1, 64, 64, 3))
        return (out_image, description or "")


NODE_CLASS_MAPPINGS = {"CSGlideVisionCS": CSGlideVision}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideVisionCS": "CSGlide Vision CS"}
