"""
csglide_preview.py -- Glide Preview

ComfyUI's Latent2RGBPreviewer renders `x0[0, :, 0]` -- the first latent frame
only -- so you watch a still image while a five second shot is sampled.

This shows all of it: every latent frame turned into a picture and either
animated on the node or tiled into a contact sheet, so motion across the whole
shot is visible while it denoises.

Two ways to turn a latent frame into a picture:

  latent2rgb   H3's own latent_rgb_factors, one matmul per frame. Free, but
               it is a colour approximation, not a decode.
  TAE          a tiny autoencoder trained on H3's latent space. Real picture,
               a few milliseconds per frame. Drop taeh3.safetensors into
               ComfyUI/models/vae_approx/ and pick it in the decoder widget.

The TAE architecture is Ollin Boer Bohan's TAEHV (github.com/madebyollin/taehv,
MIT). Only the decoder is built here, with temporal upscaling switched off so
one latent frame yields exactly one preview frame, and with spatial upscaling
enabled only as far as the requested preview resolution needs.

Wire it anywhere in the MODEL path; it passes the model straight through and
only changes how previews are drawn.
"""

import base64
import inspect
import io
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import latent_preview

try:
    import comfy.patcher_extension
    from comfy.patcher_extension import WrappersMP
except Exception:
    WrappersMP = None

try:
    import folder_paths
except Exception:
    folder_paths = None

try:
    import comfy.utils
except Exception:
    comfy = None

try:
    from server import PromptServer
except Exception:
    PromptServer = None


# --------------------------------------------------------------------------
# config, set by the node each run
# --------------------------------------------------------------------------

_CONF = {
    "enabled": False,
    "mode": "animated",
    "fps": 24.0,
    "sampler_preview": False,
    "columns": 4,
    "every_n_steps": 1,
    "max_resolution": 512,
    "decoder": "latent2rgb",
}

# True only while the sampling wrapper is running. The wrapper starts after
# get_previewer has already handed out a previewer, so the previewer asks this
# on every call rather than being chosen up front -- bypass the node and the
# old path simply takes over again.
_LIVE = {"wrapper": False}

FPS = 24.0

# frames decoded per TAE call. Bounds activation memory; the first frame of
# each batch starts with empty memory, which is invisible at preview quality.
TAE_BATCH = 8


def _output_frames(latent_t):
    """H3 compresses time ~3.35x: 17k+5 output frames -> 5k+2 latent frames."""
    if latent_t <= 2:
        return 5
    return ((int(latent_t) - 2) // 5) * 17 + 5

_orig_get_previewer = None


def _is_h3(latent_format):
    """MiniMaxH3Video and MiniMaxH3AV, without importing the class."""
    for cls in type(latent_format).__mro__:
        if cls.__name__.startswith("MiniMaxH3"):
            return True
    return False


def _pick(total, want):
    """`want` frame indices spread evenly across `total`, ends included."""
    want = max(1, min(int(want), int(total)))
    if want == 1:
        return [0]
    return [round(i * (total - 1) / (want - 1)) for i in range(want)]


def _decoder_choices():
    names = ["latent2rgb"]
    if folder_paths is not None:
        try:
            names += list(folder_paths.get_filename_list("vae_approx"))
        except Exception:
            pass
    return names


# --------------------------------------------------------------------------
# TAE decoder
#
# Architecture after TAEHV by Ollin Boer Bohan (MIT). Decoder only, and only
# the base variant -- the "super" variant is a different block stack and is
# not loaded here.
# --------------------------------------------------------------------------

def _conv(n_in, n_out, **kw):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kw)


class _Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class _MemBlock(nn.Module):
    """Residual block that also sees the previous frame's activations."""

    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            _conv(n_in * 2, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out), nn.ReLU(inplace=True),
            _conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class _TGrow(nn.Module):
    """Temporal upsample. Stride 1 here: kept so weight keys line up."""

    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        _NT, C, H, W = x.shape
        return x.reshape(-1, C, H, W)


class TAEDecoder(nn.Module):
    """H3 TAE decoder, temporal upscale off, spatial upscale selectable.

    `space_upscale` is a 3-tuple of bools. Every stage disabled halves the
    decoded resolution and quarters the activation memory, which is why the
    node picks the smallest stack that still covers the requested preview
    size. Weight shapes do not change -- the base decoder upsamples with
    nn.Upsample, so a disabled stage is simply scale_factor 1.
    """

    LATENT_CHANNELS = 24
    PATCH_SIZE = 2

    def __init__(self, space_upscale=(True, True, True)):
        super().__init__()
        n_f = [256, 128, 64, 64]
        s = [2 if u else 1 for u in space_upscale]
        self.layers = nn.Sequential(
            _Clamp(), _conv(self.LATENT_CHANNELS, n_f[0]), nn.ReLU(inplace=True),
            _MemBlock(n_f[0], n_f[0]), _MemBlock(n_f[0], n_f[0]), _MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=s[0]), _TGrow(n_f[0], 1), _conv(n_f[0], n_f[1], bias=False),
            _MemBlock(n_f[1], n_f[1]), _MemBlock(n_f[1], n_f[1]), _MemBlock(n_f[1], n_f[1]),
            nn.Upsample(scale_factor=s[1]), _TGrow(n_f[1], 1), _conv(n_f[1], n_f[2], bias=False),
            _MemBlock(n_f[2], n_f[2]), _MemBlock(n_f[2], n_f[2]), _MemBlock(n_f[2], n_f[2]),
            nn.Upsample(scale_factor=s[2]), _TGrow(n_f[2], 1), _conv(n_f[2], n_f[3], bias=False),
            nn.ReLU(inplace=True), _conv(n_f[3], 3 * self.PATCH_SIZE ** 2),
        )

    # ---- weights ----

    def load(self, state_dict):
        """Take the decoder half of a TAEHV checkpoint.

        Checkpoints are trained with temporal upsampling on, so their TGrow
        convolutions emit stride*n_f channels. With upsampling off we keep the
        last n_f of them -- the last timestep's output.
        """
        want = self.layers.state_dict()
        got = {}
        for k, v in state_dict.items():
            if not k.startswith("decoder."):
                continue
            key = k[len("decoder."):]
            if key not in want:
                continue
            target = want[key]
            if v.shape != target.shape and v.ndim == target.ndim and v.shape[0] > target.shape[0]:
                v = v[-target.shape[0]:]
            got[key] = v
        missing = [k for k in want if k not in got]
        if missing:
            raise RuntimeError(
                "not a TAEHV base decoder for H3 -- %d tensors missing, first is %s"
                % (len(missing), missing[0]))
        self.layers.load_state_dict(got)

    # ---- decode ----

    def forward(self, x):
        """[T, C, H, W] latent frames -> [T, 3, H', W'] in 0..1."""
        n = x.shape[0]
        for block in self.layers:
            if isinstance(block, _MemBlock):
                # each frame sees the one before it; the first sees zeros
                past = F.pad(x.unsqueeze(0), (0, 0, 0, 0, 0, 0, 1, 0))[:, :n]
                x = block(x, past.squeeze(0))
            else:
                x = block(x)
        x = F.pixel_shuffle(x, self.PATCH_SIZE)
        return x.clamp_(0, 1)


_TAE_CACHE = {}


def _load_tae(name, device, dtype, space_upscale):
    key = (name, str(device), str(dtype), tuple(space_upscale))
    if key in _TAE_CACHE:
        return _TAE_CACHE[key]
    if folder_paths is None or comfy is None:
        raise RuntimeError("folder_paths / comfy.utils unavailable")
    path = folder_paths.get_full_path("vae_approx", name)
    if path is None:
        raise RuntimeError("%s not found in models/vae_approx" % name)
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    model = TAEDecoder(space_upscale)
    model.load(sd)
    model = model.to(device=device, dtype=dtype).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _TAE_CACHE.clear()          # only ever one live decoder
    _TAE_CACHE[key] = model
    print("[Glide Preview] TAE decoder loaded: %s (spatial %s)"
          % (name, "".join("1" if u else "0" for u in space_upscale)))
    return model


def _space_upscale_for(latent_hw, target_long_edge):
    """Smallest stack of spatial upsamples that still covers the preview size.

    pixel_shuffle already doubles, so a latent long edge of 84 gives 168 with
    every stage off, then 336, 672, 1344.
    """
    base = max(latent_hw) * TAEDecoder.PATCH_SIZE
    stages = 0
    while stages < 3 and base < max(64, int(target_long_edge)):
        base *= 2
        stages += 1
    return tuple(i < stages for i in range(3))


# --------------------------------------------------------------------------
# previewer
# --------------------------------------------------------------------------

class FilmstripPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, latent_format):
        self.factors = torch.tensor(latent_format.latent_rgb_factors, device="cpu").transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        self.bias = torch.tensor(bias, device="cpu") if bias is not None else None
        self.reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
        self.calls = 0
        self.last = None
        self.tae_failed = False     # one complaint per run, then fall back quietly
        self.sent = False           # something has been pushed to the node this run
        self.render_failed = False

    def _rgb(self, plane):
        """[C, H, W] latent -> [H, W, 3] in roughly -1..1."""
        f = self.factors.to(dtype=plane.dtype, device=plane.device)
        b = self.bias.to(dtype=plane.dtype, device=plane.device) if self.bias is not None else None
        return torch.nn.functional.linear(plane.movedim(0, -1), f, bias=b)

    def _tae(self, x, idx):
        """Decode the picked frames. Returns a list of [H, W, 3] in -1..1."""
        planes = torch.stack([x[:, t] for t in idx])            # [T, C, H, W]
        space = _space_upscale_for(planes.shape[-2:], _CONF["max_resolution"])
        dtype = torch.float16 if planes.device.type == "cuda" else torch.float32
        model = _load_tae(_CONF["decoder"], planes.device, dtype, space)

        out = []
        with torch.no_grad():
            for i in range(0, planes.shape[0], TAE_BATCH):
                batch = planes[i:i + TAE_BATCH].to(dtype)
                rgb = model(batch).float()                      # [T, 3, H, W] 0..1
                out.extend(rgb.movedim(1, -1) * 2.0 - 1.0)      # -> [H, W, 3] -1..1
        return out

    def _tiles(self, x, idx):
        if _CONF["decoder"] != "latent2rgb" and not self.tae_failed:
            try:
                return self._tae(x, idx)
            except Exception as e:
                self.tae_failed = True
                print("[Glide Preview] TAE decode failed (%s) -- using latent2rgb" % e)
        return [self._rgb(x[:, t]) for t in idx]

    def _sheet(self, tiles):
        """Tile the frames into one still."""
        cols = max(1, min(int(_CONF["columns"]), len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        blank = torch.zeros_like(tiles[0])
        strips = []
        for r in range(rows):
            row = tiles[r * cols:(r + 1) * cols]
            row = row + [blank] * (cols - len(row))
            strips.append(torch.cat(row, dim=1))    # along width
        return torch.cat(strips, dim=0)             # along height

    # ---- wrapper path ----

    def render(self, x0, latent_shapes=None):
        """Draw from inside the sampling stack.

        The wrapper calls this instead of going through ComfyUI's previewer
        slot, so accelerators that wrap sampling themselves cannot swallow the
        callback stream. Both modes land on this node's widget.
        """
        self.calls += 1
        every = max(1, int(_CONF["every_n_steps"]))
        if self.sent and (self.calls - 1) % every != 0:
            return

        if self.reshape is not None:
            try:
                x0 = self.reshape(x0)
            except Exception:
                pass
        x0 = _unpack(x0, latent_shapes)
        if x0.ndim != 5:
            return

        x = x0[0]                                   # [C, T, H, W]
        chans = self.factors.shape[1]
        if x.shape[0] > chans:
            x = x[:chans]

        total = x.shape[1]
        duration_s = _output_frames(total) / FPS
        want = int(round(_CONF["fps"] * duration_s))
        idx = _pick(total, max(1, min(total, want)))
        tiles = self._tiles(x, idx)
        if _CONF["mode"] != "animated":
            tiles = [self._sheet(tiles)]

        self._send_animation(tiles, total)
        self.sent = True

    def _still(self, x0):
        """One middle frame, latent2rgb only. For the sampler's own slot while
        the wrapper is drawing the real preview on the node."""
        if self.reshape is not None:
            try:
                x0 = self.reshape(x0)
            except Exception:
                pass
        if x0.ndim != 5:
            return latent_preview.preview_to_image(self._rgb(x0[0]))
        x = x0[0]
        chans = self.factors.shape[1]
        if x.shape[0] > chans:
            x = x[:chans]
        return latent_preview.preview_to_image(self._rgb(x[:, x.shape[1] // 2]))

    def decode_latent_to_preview(self, x0):
        self.calls += 1
        every = max(1, int(_CONF["every_n_steps"]))
        if self.last is not None and (self.calls - 1) % every != 0:
            return self.last          # throttled: reuse the last sheet

        if self.reshape is not None:
            x0 = self.reshape(x0)

        if x0.ndim != 5:
            # not a video latent after all -- behave like the stock previewer
            plane = x0[0]
            self.last = latent_preview.preview_to_image(self._rgb(plane))
            return self.last

        x = x0[0]                                   # [C, T, H, W]
        chans = self.factors.shape[1]
        if x.shape[0] > chans:
            # AV latent declares 32 channels so both streams stay whole; the
            # video stream is the first 24
            x = x[:chans]

        total = x.shape[1]
        # H3 has one latent frame per ~3.35 output frames, so the highest
        # honest playback rate is 24 / 3.35 = 7.2 fps. Asking for more just
        # takes every latent frame there is.
        duration_s = _output_frames(total) / FPS
        want = int(round(_CONF["fps"] * duration_s))
        idx = _pick(total, max(1, min(total, want)))
        tiles = self._tiles(x, idx)

        if _CONF["mode"] == "animated":
            self._send_animation(tiles, total)
            # the built-in preview slot still wants one frame: give it the middle
            self.last = latent_preview.preview_to_image(tiles[len(tiles) // 2])
            return self.last

        cols = max(1, min(int(_CONF["columns"]), len(tiles)))
        rows = math.ceil(len(tiles) / cols)
        blank = torch.zeros_like(tiles[0])
        strips = []
        for r in range(rows):
            row = tiles[r * cols:(r + 1) * cols]
            row = row + [blank] * (cols - len(row))
            strips.append(torch.cat(row, dim=1))    # along width
        grid = torch.cat(strips, dim=0)             # along height

        self.last = latent_preview.preview_to_image(grid)
        return self.last

    # ---- animated webp, pushed to the node over our own channel ----

    def _send_animation(self, tiles, latent_t):
        """ComfyUI's preview socket encodes a single JPEG or PNG, so an
        animation cannot travel that way. Encode a looping WebP here and send
        it to the node's own widget instead."""
        if PromptServer is None or not tiles:
            return
        try:
            frames = []
            long_edge = max(1, int(_CONF["max_resolution"]))
            for t in tiles:
                img = latent_preview.preview_to_image(t)
                scale = long_edge / max(img.width, img.height)
                if scale != 1.0:
                    img = img.resize((max(1, round(img.width * scale)),
                                      max(1, round(img.height * scale))),
                                     Image.BILINEAR)
                frames.append(img.convert("RGB"))

            # play for as long as the finished shot will, thinning included
            duration_s = _output_frames(latent_t) / FPS
            per_frame_ms = max(20, int(round(1000.0 * duration_s / len(frames))))

            buf = io.BytesIO()
            frames[0].save(buf, format="WEBP", save_all=True, append_images=frames[1:],
                           duration=per_frame_ms, loop=0, quality=70, method=0)
            PromptServer.instance.send_sync("csglide.preview", {
                "webp": base64.b64encode(buf.getvalue()).decode("ascii"),
                "frames": len(frames),
                "latent_frames": int(latent_t),
                "seconds": round(duration_s, 2),
                "fps": round(len(frames) / duration_s, 1) if duration_s else 0,
                "decoder": _CONF["decoder"] if not self.tae_failed else "latent2rgb",
            })
        except Exception as e:
            print("[Glide Preview] could not send animation: %s" % e)

    def decode_latent_to_preview_image(self, preview_format, x0):
        if _LIVE["wrapper"]:
            # the wrapper already drew this shot on the node; decoding it a
            # second time for the sampler slot would just double the cost
            if not _CONF["sampler_preview"]:
                return None
            return ("JPEG", self._still(x0), int(_CONF["max_resolution"]))

        img = self.decode_latent_to_preview(x0)
        if _CONF["mode"] == "animated" and not _CONF["sampler_preview"]:
            # the animation is already on this node; a still on the sampler as
            # well is just two previews of the same thing. None means "no
            # preview this update", which the progress bar handles.
            return None
        return ("JPEG", img, int(_CONF["max_resolution"]))


def _unpack(x0, latent_shapes):
    """Rebuild a video latent from a flattened pack.

    H3 samples one packed audio+video latent. Most paths hand the callback a
    normal 5D tensor, but latent_shapes is the authority on what the pack is
    meant to be, so use it whenever what arrives is not already 5D.
    """
    if x0.ndim == 5 or not latent_shapes:
        return x0
    flat = x0.reshape(x0.shape[0], -1)
    for shape in latent_shapes:
        dims = [int(d) for d in shape]
        if len(dims) < 4:
            continue
        dims = dims[-4:]                            # [C, T, H, W]
        want = 1
        for d in dims:
            want *= d
        if flat.shape[1] >= want:
            return flat[:, :want].reshape([x0.shape[0]] + dims)
    return x0


_WRAPPER_KEY = "csglide_preview"


def _outer_sample_wrapper(executor, *args, **kwargs):
    """Sits on the model patcher, so it runs INSIDE anything else that wraps
    sampling.

    ComfyUI builds its preview callback outside every wrapper. A two-pass
    accelerator -- Spectrum with offline replay, for one -- does the real
    transformer work in its first pass and replays the schedule in the second,
    so that outer callback only fires during the replay and the whole preview
    arrives at once, seconds before the run ends. Wrapping the callback here
    puts it back on the pass that is actually denoising.
    """
    if not _CONF["enabled"]:
        return executor(*args, **kwargs)

    try:
        bound = inspect.signature(executor.original).bind(*args, **kwargs)
        bound.apply_defaults()
    except Exception as e:
        print("[Glide Preview] cannot read the sampling signature (%s) -- preview off" % e)
        return executor(*args, **kwargs)

    try:
        latent_format = executor.class_obj.model_patcher.model.latent_format
    except Exception as e:
        print("[Glide Preview] no latent format on the guider (%s) -- preview off" % e)
        return executor(*args, **kwargs)

    if not (_is_h3(latent_format)
            and getattr(latent_format, "latent_rgb_factors", None) is not None):
        return executor(*args, **kwargs)

    previewer = FilmstripPreviewer(latent_format)
    inner = bound.arguments.get("callback")
    latent_shapes = bound.arguments.get("latent_shapes")

    def callback(step, x0, x, total_steps):
        try:
            previewer.render(x0, latent_shapes)
        except Exception as e:
            if not previewer.render_failed:
                previewer.render_failed = True
                print("[Glide Preview] render failed (%s) -- previews off for this run" % e)
        if inner is not None:
            return inner(step, x0, x, total_steps)

    bound.arguments["callback"] = callback
    _LIVE["wrapper"] = True
    try:
        return executor(*bound.args, **bound.kwargs)
    finally:
        _LIVE["wrapper"] = False


def _install():
    """Patch get_previewer once. Non-H3 models are handed straight back."""
    global _orig_get_previewer
    if _orig_get_previewer is not None:
        return
    _orig_get_previewer = latent_preview.get_previewer

    def patched(device, latent_format):
        if (_CONF["enabled"]
                and _is_h3(latent_format)
                and getattr(latent_format, "latent_rgb_factors", None) is not None):
            return FilmstripPreviewer(latent_format)
        return _orig_get_previewer(device, latent_format)

    latent_preview.get_previewer = patched
    print("[Glide Preview] whole-shot previews active for MiniMax H3")


# --------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------

class CSGlidePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "mode": (["animated", "contact sheet", "stock (first frame)"], {
                    "default": "animated",
                    "tooltip": "Animated plays the whole shot on this node as it "
                               "denoises. Contact sheet tiles the frames into one "
                               "still. Stock is ComfyUI's frozen first frame.",
                }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 0.5, "max": 24.0, "step": 0.1,
                    "tooltip": "Playback rate for the preview. H3 stores one latent "
                               "frame per ~3.35 output frames, so 7.2 fps is every "
                               "frame there is - anything higher simply uses them all.",
                }),
                "columns": ("INT", {
                    "default": 4, "min": 1, "max": 12,
                    "tooltip": "Tiles per row in the contact sheet.",
                }),
                "every_n_steps": ("INT", {
                    "default": 2, "min": 1, "max": 20,
                    "tooltip": "Redraw at most this often. Raise it if previewing "
                               "is costing you sampling time.",
                }),
                "max_resolution": ("INT", {
                    "default": 512, "min": 256, "max": 4096, "step": 64,
                    "tooltip": "Longest edge of the preview sent to the browser. "
                               "Also decides how far the TAE decoder upsamples, so "
                               "raising it costs decode time as well as bandwidth.",
                }),
                "sampler_preview": ("BOOLEAN", {
                    "default": False, "label_on": "on", "label_off": "off",
                    "tooltip": "Also draw ComfyUI's single-frame preview on the "
                               "sampler node. Off by default in animated mode, "
                               "since this node already shows the whole shot.",
                }),
            },
            "optional": {
                "decoder": (_decoder_choices(), {
                    "default": "latent2rgb",
                    "tooltip": "latent2rgb is H3's colour approximation - free, but "
                               "mushy. A TAE gives real picture: put taeh3.safetensors "
                               "in ComfyUI/models/vae_approx/ and pick it here.",
                }),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "CGlide"
    DESCRIPTION = ("Preview the whole H3 shot while it denoises, instead of "
                   "ComfyUI's single frozen first frame.")

    # Accept values saved by earlier versions instead of failing validation.
    _LEGACY = {"whole shot": "contact sheet"}

    @classmethod
    def VALIDATE_INPUTS(cls, mode=None, decoder=None, **kwargs):
        return True

    def patch(self, model, mode, columns, every_n_steps, max_resolution,
              fps=24.0, sampler_preview=False, decoder="latent2rgb", frames=None):
        mode = self._LEGACY.get(mode, mode)
        if mode not in ("animated", "contact sheet", "stock (first frame)"):
            print("[Glide Preview] unknown mode %r, using animated" % (mode,))
            mode = "animated"
        if decoder != "latent2rgb" and decoder not in _decoder_choices():
            print("[Glide Preview] %r is not in models/vae_approx, using latent2rgb"
                  % (decoder,))
            decoder = "latent2rgb"
        _CONF.update({
            "enabled": mode != "stock (first frame)",
            "mode": "animated" if mode == "animated" else "sheet",
            "fps": float(fps),
            "columns": int(columns),
            "every_n_steps": int(every_n_steps),
            "max_resolution": int(max_resolution),
            "sampler_preview": bool(sampler_preview),
            "decoder": decoder,
        })
        _install()

        # Registering here rather than patching only get_previewer means the
        # preview rides inside any accelerator that wraps sampling. Wrappers
        # run in registration order, outermost first, so keeping this node
        # DOWNSTREAM of an accelerator is what puts the preview inside it.
        if WrappersMP is None or not _CONF["enabled"]:
            return (model,)
        try:
            out = model.clone()
            if _WRAPPER_KEY not in out.wrappers.get(WrappersMP.OUTER_SAMPLE, {}):
                out.add_wrapper_with_key(
                    WrappersMP.OUTER_SAMPLE, _WRAPPER_KEY, _outer_sample_wrapper)
            return (out,)
        except Exception as e:
            print("[Glide Preview] could not register the sampling wrapper (%s) "
                  "-- using the preview hook instead" % e)
            return (model,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")     # settings must reach the previewer every run


NODE_CLASS_MAPPINGS = {"CSGlidePreviewCS": CSGlidePreview}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlidePreviewCS": "Glide Preview"}
