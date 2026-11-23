import os
import io
import re
import json
import base64
import asyncio
import socket
import ssl
import http.client
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

try:
    from .csglide_paths import PathRefused, safe_path
except ImportError:  # loaded outside the package (tests)
    from csglide_paths import PathRefused, safe_path
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
# Two kinds of outgoing request, two policies.
#
#   BACKEND (analyze / models / eject / auto_eject) talks to a vision server.
#   Which server is the OPERATOR's choice, never the caller's: a URL that
#   arrives in a request or a widget (POST /prompt has no login) is only used
#   when it matches the operator's configuration. Out of the box that is a
#   local Ollama:
#
#       CSGLIDE_VISION_URL   comma-separated backend base URLs.
#                            Default: http://localhost:11434,http://127.0.0.1:11434
#                            LM Studio:  set CSGLIDE_VISION_URL=http://localhost:1234
#       CSGLIDE_VISION_ALLOW_HOSTS
#                            hosts any backend URL may point at, for a vision
#                            server on another machine or a cloud API, e.g.
#                            192.168.1.50,ollama.lan,api.openai.com
#
#   Both are environment variables on purpose: a value that arrives in a
#   request is exactly what an attacker controls, so it can't be the guard.
#
#   IMAGE FETCH (load_url) pulls a picture off the public web. It never
#   touches loopback or a private address. The check is done at CONNECT
#   time: the connection resolves the host once, refuses the whole host if
#   any address is internal, and connects to the address that passed. A DNS
#   name that answers differently a moment later (DNS rebinding) gets nothing,
#   because no second lookup happens. HTTPS still verifies the certificate
#   against the host name. Redirects and the og:image URL go through the same
#   connection, so every hop is checked the same way. This fetch does not go
#   through a system HTTP proxy: a proxy would do its own lookup.
# ----------------------------------------------------------------------------
ALLOWED_SCHEMES = ("http", "https")
MAX_FETCH_BYTES = 32 * 1024 * 1024          # a picture, not a disk image
DEFAULT_BACKENDS = "http://localhost:11434,http://127.0.0.1:11434"


class UrlNotAllowed(RuntimeError):
    """The URL is refused by the guard. Message is safe to show the user."""


def _extra_allowed_hosts():
    raw = os.environ.get("CSGLIDE_VISION_ALLOW_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _url_key(url):
    """(scheme, host, port, path) for comparing URLs, or None if not http(s)."""
    try:
        parts = urlsplit((url or "").strip())
        scheme = (parts.scheme or "").lower()
        host = (parts.hostname or "").lower()
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError:
        return None
    if scheme not in ALLOWED_SCHEMES or not host:
        return None
    return scheme, host, port, (parts.path or "").rstrip("/")


def configured_backends():
    raw = os.environ.get("CSGLIDE_VISION_URL", "") or DEFAULT_BACKENDS
    return [u.strip().rstrip("/") for u in raw.split(",") if _url_key(u.strip())]


def backend_url(url):
    """The backend base URL to use. Empty -> the first configured one. A
    supplied URL is accepted only when it is one of the configured backends,
    or its host is in CSGLIDE_VISION_ALLOW_HOSTS. Raises UrlNotAllowed."""
    url = (url or "").strip().rstrip("/")
    allowed = configured_backends()
    if not url:
        if not allowed:
            raise UrlNotAllowed("No vision server is configured (CSGLIDE_VISION_URL).")
        return allowed[0]
    key = _url_key(url)
    if key is None:
        raise UrlNotAllowed("Only http and https URLs are allowed.")
    if parse_qs_present(url):
        raise UrlNotAllowed("The vision server URL can't carry a query or fragment.")
    if key in {_url_key(u) for u in allowed} or key[1] in _extra_allowed_hosts():
        return url
    raise UrlNotAllowed(
        "'%s' is not a configured vision server. Set CSGLIDE_VISION_URL=%s "
        "(or add its host to CSGLIDE_VISION_ALLOW_HOSTS) and restart ComfyUI."
        % (url, url))


def parse_qs_present(url):
    parts = urlsplit(url)
    return bool(parts.query or parts.fragment or parts.username or parts.password)


def _is_internal(ip):
    """True for anything that is not a normal public address."""
    if ip.version == 6:
        mapped = getattr(ip, "ipv4_mapped", None)
        if mapped is not None:
            ip = mapped
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_reserved or ip.is_multicast or ip.is_unspecified
        or not ip.is_global
    )


def _resolve(host, port):
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise UrlNotAllowed("Can't resolve '%s'." % host)
    out = []
    for info in infos:
        try:
            out.append(ipaddress.ip_address(info[4][0].split("%")[0]))
        except ValueError:
            continue
    if not out:
        raise UrlNotAllowed("Can't resolve '%s'." % host)
    return out


def public_address(host, port):
    """Resolve once and return the address to connect to. Every address the
    name has must be public, or the host is refused."""
    host = (host or "").lower()
    ips = _resolve(host, port)
    if host in _extra_allowed_hosts():
        return ips[0]                   # explicit opt-in by whoever runs the server
    for ip in ips:
        if _is_internal(ip):
            raise UrlNotAllowed(
                "'%s' is a private, internal or local address, so it is blocked. "
                "This loader only fetches from the public web." % host)
    return ips[0]


def check_public_url(url):
    """Scheme/host check plus an early resolve, so a refused URL says why
    instead of timing out. The binding check is the one at connect time."""
    key = _url_key(url)
    if key is None:
        raise UrlNotAllowed("Only http and https URLs are allowed.")
    public_address(key[1], key[2])
    return url


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        ip = public_address(self.host, self.port)
        self.sock = socket.create_connection(
            (str(ip), self.port), self.timeout, self.source_address)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        ip = public_address(self.host, self.port)
        sock = socket.create_connection(
            (str(ip), self.port), self.timeout, self.source_address)
        # Certificate checked against the NAME, connection made to the address
        # that passed: TLS stays verified and no second lookup happens.
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_PinnedHTTPConnection, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    def __init__(self):
        super().__init__(context=ssl.create_default_context())

    def https_open(self, req):
        return self.do_open(_PinnedHTTPSConnection, req, context=self._context)


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """Each hop is checked. For the public fetch the connection itself checks
    again; for a backend, a redirect has to stay on a configured server."""

    def __init__(self, check):
        self.check = check

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self.check(newurl)
        return urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )


def public_open(req, timeout):
    """urlopen for the public web, pinned to a checked address, no proxy."""
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    check_public_url(url)
    opener = urllib.request.OpenerDirector()
    for h in (urllib.request.ProxyHandler({}), _PinnedHTTPHandler(),
              _PinnedHTTPSHandler(), _CheckedRedirects(check_public_url),
              urllib.request.HTTPDefaultErrorHandler(),
              urllib.request.HTTPErrorProcessor()):
        opener.add_handler(h)
    return opener.open(req, timeout=timeout)


def _check_backend_hop(url):
    """A request or redirect target for a backend call: it has to be on one
    of the configured servers (same scheme, host and port) or on an allowed
    host. Raises UrlNotAllowed."""
    key = _url_key(url)
    if key is None:
        raise UrlNotAllowed("Only http and https URLs are allowed.")
    origins = {_url_key(u)[:3] for u in configured_backends()}
    if key[:3] in origins or key[1] in _extra_allowed_hosts():
        return url
    raise UrlNotAllowed("The vision server redirected to '%s', which is not a "
                        "configured server." % key[1])


def backend_open(req, timeout):
    """urlopen for a configured vision server. The request URL has already
    been built from backend_url(); the redirect handler keeps it there."""
    url = req.full_url if isinstance(req, urllib.request.Request) else req
    _check_backend_hop(url)
    opener = urllib.request.build_opener(_CheckedRedirects(_check_backend_hop))
    return opener.open(req, timeout=timeout)


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


_ANNOTATED = re.compile(r"^(.*) \[(input|output|temp)\]$")


def load_image_from_input(name):
    """An image by name, from ComfyUI's input folder ("pic.png") or from the
    folder its annotation names ("pic.png [output]"). Anything that would leave
    input/output/temp raises PathRefused; a name that is fine but missing
    returns None."""
    name = (name or "").strip()
    if not name:
        return None
    m = _ANNOTATED.match(name)
    if m:
        path = safe_path(m.group(1), bases=(m.group(2),))
    else:
        path = safe_path(name, bases=("input",))
    if path is None:
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
        with backend_open(req, 600) as resp:
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
    with backend_open(req, 600) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    return (out["choices"][0]["message"]["content"] or "").strip()


def run_analysis(image_b64, prompt, backend, url, model, api_key=""):
    prompt = (prompt or "").strip() or DEFAULT_PROMPT
    backend = (backend or "ollama").lower()
    url = backend_url(url)
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
    if isinstance(e, (UrlNotAllowed, PathRefused, RuntimeError)):
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
            data.get("url", ""),
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
    # The public-web fetcher: pinned to a checked public address. Nothing
    # about loading a reference picture needs to reach this machine or the LAN.
    with public_open(req, 60) as r:
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
        check_public_url(url)

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
        # than the one that was typed in; public_open checks it the same way.
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
    try:
        url = backend_url(request.query.get("url", ""))

        def _tags():
            endpoint = url + "/api/tags"
            req = urllib.request.Request(endpoint)
            with backend_open(req, 15) as r:
                return json.loads(_read_capped(r).decode("utf-8"))

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
        url = backend_url(data.get("url") or "")
        model = (data.get("model") or DEFAULT_MODEL).strip()

        def _unload():
            endpoint = url + "/api/generate"
            payload = {"model": model, "keep_alive": 0}
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                endpoint, data=body, headers={"Content-Type": "application/json"}
            )
            with backend_open(req, 60) as r:
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
                endpoint = backend_url(server_url) + "/api/generate"
                body = json.dumps(
                    {"model": (model or DEFAULT_MODEL), "keep_alive": 0}
                ).encode("utf-8")
                req = urllib.request.Request(
                    endpoint, data=body, headers={"Content-Type": "application/json"}
                )
                with backend_open(req, 60) as r:
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
