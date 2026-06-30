# ComfyUI-CGlide

Two custom nodes:

- **Glide Vision** — describe an image with a vision model (Ollama / OpenAI-compatible).

  (x5 time faster to generate description with "ollama run huihui_ai/qwen3.5-abliterated:2B" for example)

  Drag/drop, Browse, URL, or clipboard (wand button).

  Gear panel for backend + model,
  Copy + Eject. Outputs `image` and `description`.
- **Glide Seed** — a seed source with the random/fixed/increment/decrement selector
  as a normal widget. Wire `seed` into a sampler's seed input.

## Install
Drop the `ComfyUI-CGlide` folder into `ComfyUI/custom_nodes/` and restart.
No extra Python packages required. For Glide Vision, install Ollama and pull a
vision model (e.g. `ollama pull qwen2.5vl`).
