# ComfyUI-CGlide

Custom nodes for **MiniMax H3** video generation in ComfyUI.

Four nodes: one that builds the whole prompt and conditioning, one that lets you
watch the shot while it samples, one that writes the file, and one that joins a
continuation onto the clip it continued from.

I make short films with these. Everything here exists because I needed it, and
the defaults are what I actually use.

<img width="1045" height="1574" alt="Capture d&#39;écran 2026-08-17 221233" src="https://github.com/user-attachments/assets/d9984423-f965-470b-80c8-3faebffa079b" />


---

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/<your-user>/ComfyUI-CGlide
```

Restart ComfyUI. The nodes appear under **CGlide** in the node menu.

**Update ComfyUI while you're at it.** Honestly, half the H3 problems people
post about are an old ComfyUI, not the model. The low-VRAM path below needs
0.31.0 or newer.

### Requirements

- **PyAV** — reading reference videos and their soundtracks. Usually already
  there with ComfyUI.
- **torchaudio** — resampling reference audio.
- **ffmpeg on your PATH** — Glide Video shells out to it. Nothing else uses it.

### Models

Everything comes from the Comfy-Org MiniMax H3 repo:

<!-- TODO: exact filenames and folders -->
- the reference-to-video checkpoint (and/or first-last-frame) → `models/diffusion_models/`
- the text encoder → `models/text_encoders/`
- the video VAE → `models/vae/`
- the audio VAE → `models/vae/`

**Less VRAM:** grab the `w4a8` model from Kijai's experimental repo — about
11.8 GB instead of 21 — and the `int8_convrot` video VAE from the same place.
Same workflow, same prompts, you only swap the loaders.

Careful, this one is nasty: if your ComfyUI is too old for `w4a8` you get a
**black video with sound**. No error, nothing. If you see that, it's the
version, not you.

---

## H3 Studio

One node instead of the pile of loaders and text boxes. Nine image slots, three
video, three audio, the prompt, and the checks — all in one panel.

It does **not** touch sampling. Same sampler, same steps as a normal H3
workflow. It builds the prompt and the conditioning, that's it.

**Inputs:** `clip`, `vae`, `audio_vae`, and `first_frame` / `last_frame` for the
first-last-frame family.

**Outputs:** `positive`, `latent`, `width`, `height`, `length`, `seconds`,
`overlap_frames`, `source_video`.

The last two are for chaining — see below. Everything else you wire the way you
would with the native H3 nodes.

### What's in the panel

- **Both model families in one node.** `First / last` and `Omni references` is a
  switch at the top; the accent colour follows so you always know which one
  you're in.
- **Drag files onto slots.** Type `@` in the prompt and pick a reference to drop
  its tag in.
- **"Sent to the encoder as".** The model sees `Picture 1`, `Picture 2` and so
  on, and the number comes from *which slots are filled*, not from the slot
  number. Leave a gap and your Picture 3 is not the one you think it is. This
  strip shows you what actually goes out.
- **Trim bars with playback.** Drag anywhere on the track to slide the window
  and the picture follows, so you can hunt for the right moment. Pick a length,
  slide it, done.
- **Prompt check.** Amber when the soundscape doesn't cover the whole clip, or
  when nothing says what the mouths are doing. Those two are most of the
  mumbling problems.
- **Speech budget.** Words against time at your reading pace, so a line that
  can't fit in the beat shows up before you render it.
- **Shot timeline.** Type a shot marker and the bar splits. One colour per shot,
  the first words of each shown in the band, and dragging a boundary rewrites
  the timestamp in the prompt. Grey means nothing is scripted there — and
  unscripted time is exactly where H3 invents sound.
- **Projects.** Several clips in one file, click to switch, autosaved. A new
  clip inherits your references so you're not reloading the same five images
  every time. `Save packed` writes a `.h3proj.zip` with the media inside — my
  whole twelve-clip film is about 42 MB.

---

## Glide Preview

ComfyUI's preview shows you one frozen latent frame while it samples. So you sit
there for ten minutes looking at a still and you have no idea if the motion is
working.

Drop this between your model and the sampler and you watch the shot move as it
denoises.

- `mode` — animated, contact sheet, or stock.
- `decoder` — `latent2rgb` is free but mushy. Put `taeh3.safetensors` (from
  madebyollin) in `ComfyUI/models/vae_approx/` and pick it here for real
  picture. **Restart before it shows in the list.**
- `every_n_steps` — raise it if previewing is costing you sampling time.
- `max_resolution` — costs decode time as well as bandwidth.

Honest limit: it caps around **7.2 fps** and I can't fix that. H3 stores about
one latent frame per 3.35 output frames, so that's every frame there is. It's a
preview, not the render — but it's enough to kill a bad take at step four
instead of at minute ten. If the decoder fails it falls back to the stock
preview and your run keeps going.

---

## Glide Video

Frames and audio into one file, in one ffmpeg pass, no second file to mux.

- **Presets** from very compatible up to 4:4:4 10-bit and lossless FFV1. The
  4:4:4 10-bit ones are visibly better on H3 output — less flat colour, more
  nuance. Read the tooltip, each preset says what it's for.
- `fps` is playback rate only. Setting it wrong changes speed, it does not add
  or drop frames. H3 is 24.
- `save_metadata` embeds the prompt and workflow in the file comment. Drag-drop
  restore into ComfyUI is reliable for MKV, hit or miss for MP4.
- `fallback_on_failure` retries with a more compatible preset instead of
  throwing away a finished render. It says so loudly when it does.

---

## Glide Join — extending a clip

H3 gives you a few seconds at a time. This is how you get more.

H3 Studio's **CONTINUE FROM** slot anchors the tail of your previous clip, so
the model continues the motion and the sound instead of starting fresh. Glide
Join then assembles the result onto the source.

**Wiring:**

```
H3 Studio ──positive──> sampler ──> VAE Decode ──images──> Glide Join ──> Glide Video
          ──overlap_frames──────────────────────────────>
          ──source_video────────────────────────────────>
                                    VAE Decode Audio ──audio──>
```

Wiring `overlap_frames` and `source_video` across means the window length is
typed in **one place**. Change the window in the node and the join follows.

**Widgets:**

- `seam_mode` — `early_cut` keeps the continuation's version of the overlap and
  colours it onto the source. One continuous model trajectory, so motion carries
  best. `early_scurve` is the same join ramped over a few frames, for when the
  speed change at the join is visible. `hard_cut` keeps the source's own frames
  instead.
- `seam_blend_frames` — ramp length, **only read by `early_scurve`**.
- `match_levels` — the model re-renders the overlap in its own colour, so this
  fits gain and offset per frame against the source and holds the last fit
  afterwards. Leave it on.
- `match_tail` — H3 rounds its audio grid up, so every clip carries about 8 ms
  more sound than picture. That grows at every join. Leave it on.

**Leave the CONTINUE FROM slot empty and the whole thing is a normal
generation** — H3 Studio sends nothing, Glide Join passes the frames straight
through. One workflow for both, no rewiring, no bypassing.

Two things worth knowing:

- The window is **locked to the tail** of the clip, on purpose. A window from
  the middle produces material that follows on from the middle, and splicing
  that after the last frame skips everything between — it reads as a hard cut
  and no seam mode can fix it.
- Colour drift down a chain is real but small when the window is a true tail. If
  you see a step at the join it's more likely to be motion than colour.

---

## Careful with this

- **A `.py` change needs a full ComfyUI restart**, not a browser refresh.
- **If a node's widget list changes shape between versions, delete and re-add
  the node.** ComfyUI matches saved widget values by position and will smear old
  values across new widgets.
- The node is new. There will be bugs — open an issue and tell me what you did,
  that's genuinely more useful than a star.

---

## Credits

- MiniMax for H3, and the ComfyUI team for the core H3 nodes this builds on.
- **madebyollin** for the TAE decoder that makes the preview readable.
- **Kijai** for the quantised models that put this on smaller cards.
