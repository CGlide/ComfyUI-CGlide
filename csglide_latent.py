"""Glide Latent Load -- bring a saved take back as a LATENT, to upscale it.

Glide Video with save_latent = full writes <name>.h3full beside the video: the
whole latent the first sampler made, never decoded. This node lists those
files and hands one back as an ordinary H3 audio+video LATENT, so an upscale
workflow can start from the real sampler output instead of re-running the
first pass - or re-encoding the video, which costs exactly the round trip
latent hold was built to avoid.

Conditioning is not in the file. Open the same clip in H3 Studio and wire its
positive_refine into the refine sampler: the project still holds the clip's
prompt and references.
"""

import os

import torch

import comfy.model_management
import comfy.nested_tensor
import folder_paths

FULL_SIDECAR = ".h3full"
LIST_LIMIT = 500          # newest first; an output folder can hold thousands


def _output_root():
    return os.path.realpath(folder_paths.get_output_directory())


def _inside(root, target):
    root = os.path.normcase(os.path.realpath(root))
    target = os.path.normcase(os.path.realpath(target))
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:
        return False


def _list_full_latents():
    """Every .h3full under output/, as output-relative paths, newest first."""
    root = _output_root()
    found = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(FULL_SIDECAR):
                full = os.path.join(dirpath, f)
                try:
                    found.append((os.path.getmtime(full),
                                  os.path.relpath(full, root).replace("\\", "/")))
                except OSError:
                    pass
    found.sort(reverse=True)
    return [rel for _t, rel in found[:LIST_LIMIT]]


def _resolve(rel):
    """An output-relative name from the dropdown -> an absolute path, or None.
    The value comes from the client, so it is checked, never trusted."""
    if not isinstance(rel, str) or not rel.endswith(FULL_SIDECAR):
        return None
    root = _output_root()
    path = os.path.realpath(os.path.join(root, rel))
    if not _inside(root, path) or not os.path.isfile(path):
        return None
    return path


def _pictures(latent_t):
    """Latent frames -> picture frames on H3's grid (2 head + 5 per 17)."""
    return 5 if latent_t <= 2 else ((latent_t - 2) // 5) * 17 + 5


class CSGlideLatentLoad:
    @classmethod
    def INPUT_TYPES(cls):
        files = _list_full_latents()
        return {
            "required": {
                "latent_file": (files or ["(no .h3full in output - set save_latent "
                                          "to full on Glide Video)"], {
                    "tooltip": "A full latent saved by Glide Video (save_latent = "
                               "full), newest first. Press R to refresh the list "
                               "after a render."}),
            },
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("latent", "width", "height", "length")
    FUNCTION = "load"
    CATEGORY = "CGlide"
    DESCRIPTION = ("Load a take's full latent, saved by Glide Video, to upscale it "
                   "without re-running the first pass.")

    def load(self, latent_file):
        path = _resolve(latent_file)
        if path is None:
            raise FileNotFoundError(
                "Glide Latent Load: %r is not a .h3full in the output folder. Set "
                "save_latent to full on Glide Video, render, then press R to "
                "refresh the list." % latent_file)
        from safetensors.torch import load_file
        saved = load_file(path)
        video = saved.get("video")
        if video is None or video.dim() != 5:
            raise ValueError("Glide Latent Load: %s holds no H3 video latent."
                             % os.path.basename(path))
        dev = comfy.model_management.intermediate_device()
        video = video.to(dev)
        audio = saved.get("audio")
        if audio is not None:
            samples = comfy.nested_tensor.NestedTensor((video, audio.to(dev)))
        else:
            print("[Glide Latent Load] %s has no audio latent - video only"
                  % os.path.basename(path))
            samples = video
        width, height = int(video.shape[4]) * 16, int(video.shape[3]) * 16
        length = _pictures(int(video.shape[2]))
        print("[Glide Latent Load] %s -> %dx%d, %d frames"
              % (latent_file, width, height, length))
        return ({"samples": samples}, width, height, length)

    @classmethod
    def IS_CHANGED(cls, latent_file):
        path = _resolve(latent_file)
        try:
            return "%s:%d" % (path, os.stat(path).st_mtime_ns) if path else ""
        except OSError:
            return ""

    @classmethod
    def VALIDATE_INPUTS(cls, latent_file):
        # a file rendered after the list was built is still valid
        return True if _resolve(latent_file) else \
            "not a .h3full in the output folder: %s" % latent_file


NODE_CLASS_MAPPINGS = {"CSGlideLatentLoadCS": CSGlideLatentLoad}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideLatentLoadCS": "Glide Latent Load"}
