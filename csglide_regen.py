"""
Glide H3 LoRA (Diffusers) -- load a MiniMax H3 LoRA saved with Diffusers module
names onto ComfyUI's native H3 model.

Why this exists
---------------
NVlabs' Open-H3-Regenerate-2K ships its refiner as ONE merged LoRA (refiner +
LightX2V FL2VA Turbo + LMS Sharp, strengths folded in, scale 1) in the Diffusers
layout: transformer_blocks.N.attn.to_q / to_k / to_v, ff.net.0.proj, ...
ComfyUI's H3 only maps LoRAs written in the ORIGINAL layout (blocks.N.attn.qkv_proj,
mlp.fc1, ...), so the stock LoRA loader skips every key of that file and the model
runs as if no LoRA were loaded - silently.

The two layouts differ in two places that matter:
  * q, k and v are three Linears in Diffusers and ONE fused qkv_proj here
    (q | k | v along the output rows);
  * the SwiGLU input is ff.net.0.proj = [value | gate] in Diffusers and
    mlp.fc1 = [gate | value] here (see NVlabs tools/merge_adapters.py, which
    goes the other way).
Each Diffusers LoRA is therefore patched onto its exact slice of the fused weight
with ComfyUI's offset patches - nothing is merged, nothing is converted on disk,
and the result equals the Diffusers runtime's delta.

The LMS Sharp "curve" branches (sharp_curve_branches.safetensors, optional) are
LoRAs on each block's adaln_proj.linear in its CURVE form, an 8-wide input read
from adaln_t_table. Only curve-form checkpoints (the Comfy-Org pruned ones and
models built from them) have that layer; on any other base they are skipped
with a message, never forced onto a weight of the wrong shape.
"""

from __future__ import annotations

import os
import re

import torch

import comfy.lora
import folder_paths

try:
    from safetensors import safe_open
except Exception:  # pragma: no cover
    safe_open = None


_QKV_RE = re.compile(r"^(.*)\.attn\.to_([qkv])$")
# The merged Open-H3-Regenerate-2K LoRA concatenates its three adapters along
# the rank axis, in this order (NVlabs tools/merge_adapters.py, module_ranks.json):
# refiner r384, LightX2V FL2VA Turbo r128, LMS Sharp r64 = 576 on every module.
# Slicing the rank axis gives each adapter back exactly, so each can have its
# own strength - as narrow() views, nothing is copied.
PARTS = (("refiner", 384), ("turbo", 128), ("sharp", 64))
_CACHE = {}          # one entry: (path, mtime, size, precision) -> {key: tensor}


def _comfy_name(name: str) -> str:
    """Diffusers H3 module name -> ComfyUI H3 module name (inverse of the
    mapping in NVlabs' merge_adapters.convert_external)."""
    n = name
    if n.startswith("transformer."):
        n = n[len("transformer."):]
    n = n.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.")
    if n.startswith("transformer_blocks."):
        n = "blocks." + n[len("transformer_blocks."):]
    n = n.replace("norm_out.linear", "final_layer.adaln_proj.linear")
    n = n.replace(".attn.to_out.0", ".attn.out_proj")
    n = n.replace(".ff.net.2", ".mlp.fc2")
    return n


def _load(path: str, precision: str) -> dict:
    """The file as {key: tensor}, cast tensor by tensor so a 12 GB FP32 file
    never sits in RAM twice. The last file loaded is kept, so re-running the
    graph does not read it again."""
    st = os.stat(path)
    key = (path, st.st_mtime, st.st_size, precision)
    if key in _CACHE:
        return _CACHE[key]
    _CACHE.clear()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    out = {}
    if safe_open is not None and path.lower().endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                out[k] = t.to(dtype) if t.is_floating_point() else t
    else:
        import comfy.utils
        raw = comfy.utils.load_torch_file(path, safe_load=True)
        for k in list(raw.keys()):
            t = raw.pop(k)
            out[k] = t.to(dtype) if t.is_floating_point() else t
    _CACHE[key] = out
    return out


def _pairs(sd: dict, a_suffix: str, b_suffix: str):
    """(root, A, B) for every factor pair in the file."""
    for k in sd:
        if k.endswith(a_suffix):
            root = k[:-len(a_suffix)]
            b = sd.get(root + b_suffix)
            if b is not None:
                yield root, sd[k], b


def build_patch_sources(sd: dict, model_keys: set, split=True):
    """Rewrite a Diffusers-layout H3 LoRA into per-adapter groups for
    comfy.lora.load_lora: {group: (lora_dict, to_load)}. Every source pair
    becomes one or two pseudo-entries per group, pointing at a ComfyUI weight,
    sliced with an offset where the layers are fused. With split, a pair whose
    rank is the merged 576 is cut into refiner / turbo / sharp; anything else
    lands in group "all". Returns (groups, stats)."""
    groups = {}
    stats = {"pairs": 0, "patches": 0, "missing": [], "native": 0, "split": 0, "whole": 0}
    total = sum(r for _, r in PARTS)
    i = 0

    def put(target, A, B, offset=None):
        nonlocal i
        if split and A.shape[0] == total and B.shape[1] == total:
            pieces, o = [], 0
            for name, r in PARTS:
                pieces.append((name, A.narrow(0, o, r), B.narrow(1, o, r)))
                o += r
            stats["split"] += 1
        else:
            pieces = [("all", A, B)]
            stats["whole"] += 1
        for g, a, b in pieces:
            lora, to_load = groups.setdefault(g, ({}, {}))
            p = "cglide_regen.%d" % i
            i += 1
            lora[p + ".lora_A.weight"] = a
            lora[p + ".lora_B.weight"] = b
            to_load[p] = target if offset is None else (target, offset)
        stats["patches"] += 1

    for root, A, B in _pairs(sd, ".lora_A.weight", ".lora_B.weight"):
        stats["pairs"] += 1
        name = root
        if name.startswith("diffusion_model."):
            # already ComfyUI layout: pass straight through
            key = name + ".weight"
            if key in model_keys:
                put(key, A, B)
                stats["native"] += 1
            else:
                stats["missing"].append(name)
            continue

        m = _QKV_RE.match(name)
        if m:
            key = "diffusion_model." + _comfy_name(m.group(1)) + ".attn.qkv_proj.weight"
            if key not in model_keys:
                stats["missing"].append(name)
                continue
            size = int(B.shape[0])
            off = {"q": 0, "k": 1, "v": 2}[m.group(2)] * size
            put(key, A, B, (0, off, size))
            continue

        if name.endswith(".ff.net.0.proj"):
            base = name[:-len(".ff.net.0.proj")]
            key = "diffusion_model." + _comfy_name(base) + ".mlp.fc1.weight"
            if key not in model_keys:
                stats["missing"].append(name)
                continue
            half = int(B.shape[0]) // 2
            # Diffusers [value | gate]  ->  ComfyUI fc1 [gate | value]
            put(key, A, B[:half], (0, half, half))     # value rows
            put(key, A, B[half:], (0, 0, half))        # gate rows
            continue

        key = "diffusion_model." + _comfy_name(name) + ".weight"
        if key in model_keys:
            put(key, A, B)
        else:
            stats["missing"].append(name)

    return groups, stats


def build_curve_sources(sd: dict, model_keys: set):
    """LMS Sharp curve AdaLN branches -> patches on blocks.N.adaln_proj.linear."""
    lora, to_load = {}, {}
    n = 0
    missing = []
    for root, A, B in _pairs(sd, ".curve_A.weight", ".curve_B.weight"):
        key = "diffusion_model." + _comfy_name(root) + ".weight"
        if key not in model_keys:
            missing.append(root)
            continue
        p = "cglide_curve.%d" % n
        n += 1
        lora[p + ".lora_A.weight"] = A
        lora[p + ".lora_B.weight"] = B
        to_load[p] = key
    return lora, to_load, n, missing


def _curve_form(model):
    """(is_curve_form, adaln input width) of the H3 diffusion model."""
    try:
        dm = model.model.diffusion_model
        width = int(dm.blocks[0].adaln_proj.linear.in_features)
        return bool(getattr(dm, "use_adaln_curves", False)), width
    except Exception:
        return False, None


class CSGlideH3DiffLora:
    @classmethod
    def INPUT_TYPES(cls):
        loras = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (loras, {
                    "tooltip": "A MiniMax H3 LoRA saved with DIFFUSERS module names - e.g. "
                               "open_h3_regenerate_2k_lora.safetensors from "
                               "Efficient-Large-Model/Open-H3-Regenerate-2K (lora/). The stock "
                               "LoRA loader silently ignores these."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05,
                                       "tooltip": "Master: multiplies all three sliders below. 1.0 with "
                                                  "the three at 1.0 is the released recipe."}),
                "curve_name": (["none"] + loras, {
                    "tooltip": "Optional: sharp_curve_branches.safetensors (same release). The LMS "
                               "Sharp part that rides the timestep curve. Applied only on curve-form "
                               "H3 checkpoints (Comfy-Org pruned and models built from them); "
                               "skipped with a console note otherwise."}),
                "precision": (["bf16", "fp32"], {
                    "default": "bf16",
                    "tooltip": "How the LoRA is held in RAM. The file is FP32 (12.4 GB); bf16 halves "
                               "that with no visible difference - ComfyUI applies the delta in "
                               "float32 either way."}),
                "refiner": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05,
                                      "tooltip": "Open-H3 refiner (rank 384): the part that rebuilds detail "
                                                 "at the new size. 1.0 = as released. Relative to the strength "
                                                 "NVlabs folded into the file."}),
                "turbo": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05,
                                    "tooltip": "LightX2V FL2VA Turbo (rank 128): what makes 3 steps enough. "
                                               "Lower it and the refine gets softer unless you add steps. "
                                               "1.0 = as released."}),
                "sharp": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 3.0, "step": 0.05,
                                    "tooltip": "LMS Sharp (rank 64, plus its curve branches when loaded): "
                                               "edge and texture crispness. 1.0 = as released."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "CGlide"
    DESCRIPTION = ("Load a Diffusers-layout MiniMax H3 LoRA (Open-H3-Regenerate-2K) onto ComfyUI's "
                   "native H3 model, with q/k/v and the SwiGLU halves mapped onto the fused layers.")

    def load(self, model, lora_name, strength, curve_name, precision,
             refiner=1.0, turbo=1.0, sharp=1.0):
        master = strength
        strength = {"refiner": master * refiner, "turbo": master * turbo,
                    "sharp": master * sharp, "all": master}
        if master == 0 or (refiner == 0 and turbo == 0 and sharp == 0):
            return (model,)
        model_keys = set(model.model.state_dict().keys())

        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        sd = _load(path, precision)
        groups, stats = build_patch_sources(sd, model_keys)
        if not stats["patches"]:
            raise ValueError(
                "Glide H3 LoRA: nothing in %s maps onto this model (%d factor pairs read). Is it a "
                "MiniMax H3 LoRA, and is the model an H3 checkpoint?" % (lora_name, stats["pairs"]))

        curve_note = ""
        if curve_name != "none":
            is_curve, width = _curve_form(model)
            csd = _load_curve(folder_paths.get_full_path_or_raise("loras", curve_name), precision)
            sample = next((t for k, t in csd.items() if k.endswith(".curve_A.weight")), None)
            need = int(sample.shape[1]) if sample is not None else None
            if not is_curve or (need is not None and width != need):
                curve_note = (", curve branches SKIPPED (this checkpoint is not curve-form: its "
                              "adaln input is %s wide, the branches need %s)" % (width, need))
            else:
                c_lora, c_load, c_n, c_missing = build_curve_sources(csd, model_keys)
                g = groups.setdefault("sharp", ({}, {}))     # the curve branches are Sharp's
                g[0].update(c_lora)
                g[1].update(c_load)
                curve_note = ", %d curve branches" % c_n
                if c_missing:
                    curve_note += " (%d did not map)" % len(c_missing)

        out = model.clone()
        for g, (lora, to_load) in groups.items():
            if strength[g] == 0 or not to_load:
                continue
            out.add_patches(comfy.lora.load_lora(lora, to_load, log_missing=False), strength[g])

        mix = (" (refiner %.2f, turbo %.2f, sharp %.2f)"
               % (strength["refiner"], strength["turbo"], strength["sharp"])) if stats["split"] else ""
        if stats["whole"] and stats["split"]:
            mix += ", %d modules not in the 384/128/64 layout taken whole" % stats["whole"]
        elif stats["whole"] and not stats["split"]:
            mix = (" - not the merged Open-H3 layout: loaded whole at the master strength (refiner/turbo sliders "
                   "do not apply%s)" % (", sharp still scales the curve branches" if "sharp" in groups else ""))
        print("[Glide H3 LoRA] %s: %d modules -> %d patches%s%s%s"
              % (lora_name, stats["pairs"] - len(stats["missing"]), stats["patches"], mix, curve_note,
                 (", %d did not map (first: %s)" % (len(stats["missing"]), stats["missing"][0]))
                 if stats["missing"] else ""))
        return (out,)


def _load_curve(path, precision):
    # the curve file is small (1.2 GB) and sits beside the main one; kept out of
    # the single-entry cache so loading it does not evict the big file
    dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    out = {}
    if safe_open is not None and path.lower().endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                out[k] = t.to(dtype) if t.is_floating_point() else t
        return out
    import comfy.utils
    raw = comfy.utils.load_torch_file(path, safe_load=True)
    return {k: (t.to(dtype) if t.is_floating_point() else t) for k, t in raw.items()}


NODE_CLASS_MAPPINGS = {"CSGlideH3DiffLoraCS": CSGlideH3DiffLora}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideH3DiffLoraCS": "Glide H3 LoRA (Diffusers)"}
