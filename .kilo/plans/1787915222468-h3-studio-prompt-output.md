# Plan: Add a `prompt` STRING output to the H3 Studio node (CSGlideCast)

## Context
`CSGlideCast` (in `csglide_cast.py`, registered as `CSGlideCastCS` / display "H3 Studio") builds a
final prompt string inside `build()` but currently discards it after feeding it to the CLIP tokenizer.
The user wants a new output edge to expose that prompt so it can be wired onward (e.g. into a text
preview / debug node).

Output slots in ComfyUI are auto-generated from `RETURN_TYPES` / `RETURN_NAMES` on the Python class.
The custom frontend `web/csglide_cast.js` does **not** override output-slot definitions (its "output"
references are all about the render output *folder*, not node outputs), so this change is Python-only.

## Decision
- Emit the **final transcribed prompt** — the exact string sent to `clip.tokenize(...)`, i.e. after
  `@token` → `<Picture>/<Video>/<Audio>` substitution and `inject_carry_note` injection. This is what
  the model actually sees, and is what "the prompt this node creates" means.
- Append the new output at the **end** of the return tuple so existing workflows / links (which
  reference outputs by index) keep working — purely additive, no breakage.

## Changes — `csglide_cast.py` only

### 1. `RETURN_TYPES` / `RETURN_NAMES` (class `CSGlideCast`, ~lines 755-758)
Append `"STRING"` and `"prompt"`:

```python
RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT",
                "STRING", "IMAGE", "STRING")
RETURN_NAMES = ("positive", "latent", "width", "height", "length", "seconds",
                "overlap_frames", "source_video", "guide_frames", "prompt")
```

### 2. `finish` closure in `build()` (~line 952)
Thread the already-built `prompt` through and append it to the return tuple:

```python
def finish(cond, prompt):
    """Shared tail for both modes: attach the continuation, if any."""
    guide, overlap, guide_frames = self._continuation(
        cfg, vae, audio_vae, latent, width, height, frame_count)
    if guide_frames is None:
        guide_frames = torch.zeros((1, 64, 64, 3))
    if guide is not None:
        keyframes = list(cond[0][1].get("minimax_keyframes", []))
        keyframes.append(guide)
        cond = node_helpers.conditioning_set_values(
            cond, {"minimax_keyframes": keyframes})
    source = ""
    if cfg.get("cont"):
        source = _resolve_asset(cfg["cont"]["file"]) or cfg["cont"]["file"]
    return (cond, latent, width, height, frame_count, seconds, overlap,
            source, guide_frames, prompt)
```

### 3. Both call sites of `finish` (fl2va ~line 1004, ref2va ~line 1022)
Pass the local `prompt` (already assigned in each branch before the call):

- fl2va: `return finish(cond, prompt)`
- ref2va: `return finish(cond, prompt)`

`prompt` is already defined in both branches (`transcribe_prompt(...)` / `transcribe_prompt(inject_carry_note(cfg), tags, known)`), so no new computation is needed.

## Out of scope
- `web/csglide_cast.js` — no output-slot logic to update.
- `__init__.py` — node registration unchanged.
- `example_workflows/Minimax_h3_multi.json` — additive output at the tail keeps existing links valid.
- `IS_CHANGED` — unaffected.

## Validation
1. Reload ComfyUI; confirm the H3 Studio node now shows a `prompt` output slot (STRING) after
   `guide_frames`.
2. Run a minimal graph in each mode (fl2va and ref2va) and connect `prompt` → a "Show Text" / debug
   string node. Confirm the text shows `@token`s resolved to `<Picture n>` / `<Video k>` / `<Audio j>`
   and, in ref2va with a carry slot, the prepended carry note.
3. Load an existing saved workflow that uses H3 Studio and confirm prior links (positive, latent,
   width, etc.) still connect correctly — no shift from the appended output.
4. Optionally: `python -c "import csglide_cast"` from the custom_nodes dir to catch syntax errors
   before launching ComfyUI.
