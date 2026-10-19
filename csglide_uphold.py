"""Glide Upscale Hold -- latent hold, at upscale size.

In H3 Studio's upscale pass every clip is refined on its own, so at a join the
two refines each invent their own fine texture and the seam jumps. This pins
the head of the clip being refined to the REFINED tail of the clip before it,
the same way latent hold pins the draft chain: the held rows are re-injected
clean every step, the rest is refined around them. The refine then continues
the previous clip's texture instead of starting its own.

Wiring, in the upscale workflow:

    Concat AV Latent -> Glide Upscale Hold -> refine sampler (latent_image)
    H3 Studio  source_video / overlap_frames -> Glide Upscale Hold
    refine sampler output -> Glide Video latent   (save_latent = tail)

The last line is what makes the next link work: it saves the refined tail
beside each upscaled render, and this node reads it on the following clip.
No source (the first clip, or a cut) and it passes the latent through.
"""

import os

import torch

import comfy.nested_tensor

LATENT_SIDECAR = ".h3latent"


def _latent_t(frames):
    """Pictures -> latent frames on H3's grid (2 head + 5 per 17)."""
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def _snap(frames):
    """Down to the grid (17k+5) - the same snap the continuation uses."""
    n = int(frames)
    while n > 5 and n % 17 != 5:
        n -= 1
    return max(5, n)


class CSGlideUpscaleHold:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {
                    "tooltip": "The upscaled audio+video latent, right before the "
                               "refine sampler (Concat AV Latent's output)."}),
                "source_video": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": "H3 Studio's source_video. In the upscale pass that "
                               "is the previous clip's UPSCALED render; its refined "
                               "tail (.h3latent) is read from beside it. Empty: "
                               "pass through."}),
                "overlap_frames": ("INT", {
                    "default": 22, "min": 0, "max": 4096, "forceInput": True,
                    "tooltip": "H3 Studio's overlap_frames - how much of the clip "
                               "before this one re-renders here."}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "hold"
    CATEGORY = "CGlide"
    DESCRIPTION = ("Pin the head of an upscaled clip to the refined tail of the clip "
                   "before it, so a chain refined clip by clip joins without a "
                   "texture jump.")

    def hold(self, latent, source_video, overlap_frames):
        src = (source_video or "").strip().strip('"')
        ov = int(overlap_frames or 0)
        if not src or ov <= 0:
            return (latent,)

        side = os.path.splitext(src)[0] + LATENT_SIDECAR
        if not os.path.isfile(side):
            print("[Glide Upscale Hold] no refined tail beside %s - wire the refine "
                  "sampler's output into Glide Video's latent (save_latent = tail) "
                  "so the next clip can hold it. Passing through."
                  % os.path.basename(src))
            return (latent,)

        samples = latent.get("samples")
        if hasattr(samples, "unbind"):
            parts = samples.unbind()
        elif hasattr(samples, "tensors"):
            parts = samples.tensors
        else:
            parts = (samples,)
        video = parts[0]
        audio = parts[1] if len(parts) > 1 else None
        if video.dim() != 5:
            print("[Glide Upscale Hold] not an H3 video latent - passing through")
            return (latent,)

        from safetensors.torch import load_file
        try:
            prev = load_file(side).get("video")
        except Exception as e:
            print("[Glide Upscale Hold] could not read %s (%s) - passing through"
                  % (os.path.basename(side), e))
            return (latent,)
        if prev is None or prev.dim() != 5:
            print("[Glide Upscale Hold] %s holds no video latent - passing through"
                  % os.path.basename(side))
            return (latent,)
        if tuple(prev.shape[1:2]) + tuple(prev.shape[3:]) != \
                tuple(video.shape[1:2]) + tuple(video.shape[3:]):
            print("[Glide Upscale Hold] size mismatch: previous tail is %dx%d latent, "
                  "this clip is %dx%d - was the previous clip upscaled with the same "
                  "settings? Passing through."
                  % (prev.shape[4], prev.shape[3], video.shape[4], video.shape[3]))
            return (latent,)

        n = _snap(ov)
        k = _latent_t(n)
        if k > prev.shape[2] or k >= video.shape[2]:
            print("[Glide Upscale Hold] %d latent frames do not fit (previous %d, this "
                  "clip %d) - passing through" % (k, prev.shape[2], video.shape[2]))
            return (latent,)

        new_video = video.clone()
        new_video[:, :, :k] = prev[:, :, -k:].to(device=video.device, dtype=video.dtype) \
            .expand(video.shape[0], -1, -1, -1, -1)
        v_mask = torch.ones_like(video)
        v_mask[:, :, :k] = 0.0

        out = dict(latent)
        old_mask = latent.get("noise_mask")
        if audio is not None:
            if old_mask is not None and hasattr(old_mask, "unbind"):
                a_mask = old_mask.unbind()[1]
            else:
                a_mask = torch.ones_like(audio)
            out["samples"] = comfy.nested_tensor.NestedTensor((new_video, audio))
            out["noise_mask"] = comfy.nested_tensor.NestedTensor((v_mask, a_mask))
        else:
            out["samples"] = new_video
            out["noise_mask"] = v_mask
        print("[Glide Upscale Hold] held %d latent frames (%d pictures) from %s"
              % (k, n, os.path.basename(side)))
        return (out,)


NODE_CLASS_MAPPINGS = {"CSGlideUpscaleHoldCS": CSGlideUpscaleHold}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideUpscaleHoldCS": "Glide Upscale Hold"}
