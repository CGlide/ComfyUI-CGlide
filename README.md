# ComfyUI-CGlide

Custom nodes for **MiniMax H3** video generation in ComfyUI.

The main one, **H3 Studio**, builds the whole prompt and conditioning, and runs
a whole project of clips: a timeline, links between clips, a song across all of
them, and one Render button. Around it: a node to watch the shot while it
samples, one to write the file, one to join a continuation onto the clip it
continued from, and one to lock real audio into the latent.

I make short films with these. Everything here exists because I needed it, and
the defaults are what I actually use.
<p align="center">
<a href="https://youtu.be/aEUM6sihSdI">
<img width="470" height="758" alt="Capture d&#39;écran 2026-10-17 164418" src="https://github.com/user-attachments/assets/55793f48-b879-4a99-82ff-ce2ff1c14918" />



</a>
<br>
<em>THE RECITATION — a short film made with these nodes</em>

</p align="center">


<p align="center">
<a href="https://youtu.be/aEUM6sihSdI">
<img width="500" height="750" alt="Capture d&#39;écran 2026-08-17 221233" src="https://github.com/user-attachments/assets/d9984423-f965-470b-80c8-3faebffa079b" />
</a>
<br>
<em>THE RECITATION — a short film made with these nodes</em>

</p align="center">

---

## What's new 😎

- **Latent hold** — chained clips stay clean. Six links in a row and the
  colour, contrast and detail stay flat, where the old way went duller and
  crunchier at every join. [More below.](#latent-hold--chains-that-dont-cook)
- **Project timeline with links** — 🔗 between two clips means "this one
  continues that one". One Render button, pick what to render.
- **Song mode** — one track over the whole project. Each clip locks its own
  window of it, so H3 animates against the real song. Lay out a whole chain
  from the song in one click.
- **Timeline audio** — place audio *in* the clip at a time, H3 fills the gaps.
  Play it, scrub it, drop files straight on it.
- **Glide Audio Lock** — the same audio lock as a standalone node.

---

## Install

Search for **CGlide** in ComfyUI Manager and pick **nightly**, or:

```
cd ComfyUI/custom_nodes
git clone https://github.com/CGlide/ComfyUI-CGlide
```

Restart ComfyUI. The nodes appear under **CGlide** in the node menu.

**Updating:** if you cloned it (or installed `nightly`), `git pull` inside the
`ComfyUI-CGlide` folder, or Manager → Update. Then restart and hard-refresh the
browser (Ctrl+F5).

Careful: the Manager's registry version installs into a folder called `cglide`,
not `ComfyUI-CGlide`, and it's not a git clone, so `git pull` does nothing
there. If you have a `cglide` folder, delete it and clone instead. Don't keep
both, the pack loads twice.

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
video, three audio, the prompt, the checks, and the whole project — all in one
panel.

It does **not** touch sampling. Same sampler, same steps as a normal H3
workflow. It builds the prompt and the conditioning, that's it.

**Inputs:** `clip`, `vae`, `audio_vae`, and `first_frame` / `last_frame` for the
first-last-frame family.

**Outputs:** `positive`, `latent`, `width`, `height`, `length`, `seconds`,
`overlap_frames`, `source_video`, `guide_frames`, `positive_refine`,
`clip_name`, `locked_audio`.

`overlap_frames`, `source_video` and `locked_audio` go to Glide Join, see below.
`positive_refine` is for a second refine pass after a latent upscale (same
prompt, no continuation anchor, bigger references if you set refine above 1×).
Everything else you wire the way you would with the native H3 nodes.

### What's in the panel

- **Both model families in one node.** `First / last` and `Omni references` is a
  switch at the top; the accent colour follows so you always know which one
  you're in.
- **Drag files onto slots.** Or paste — Ctrl+V drops a screenshot straight into
  the slot under your pointer. Type `@` in the prompt and pick a reference to
  drop its tag in.
- **Several images at once.** The nine image slots take a whole batch, from the
  picker or from a drop. Drop onto a particular card and the batch fills from
  that card forward; drop onto the rack and it fills whatever slots are empty
  and leaves the filled ones alone.
- **Drag a picture onto another slot to swap them.** The `@tags` travel with the
  picture — move `@image1` to slot 3 and the prompt is rewritten to `@image3`,
  so every sentence you wrote about it still describes it. Hold **Alt** while
  releasing to move the picture and leave the prompt alone.
- **"Sent to the encoder as".** The model sees `Picture 1`, `Picture 2` and so
  on, and the number comes from *which slots are filled*, not from the slot
  number. Leave a gap and your Picture 3 is not the one you think it is. This
  strip shows you what actually goes out.
- **Trim bars with playback.** Drag anywhere on the track to slide the window
  and the picture follows, so you can hunt for the right moment.
- **Prompt check.** Amber when the soundscape doesn't cover the whole clip, or
  when nothing says what the mouths are doing. Those two are most of the
  mumbling problems.
- **Shot timeline.** Type a shot marker and the bar splits. One colour per shot,
  the first words of each shown in the band, and dragging a boundary rewrites
  the timestamp in the prompt. Grey means nothing is scripted there — and
  unscripted time is exactly where H3 invents sound.

---

## Projects — the timeline

Several clips in one file, autosaved. Open the **Project** panel (the camera
button) to see them all; the timeline strip at the top of the node shows them
in order with their lengths. Click a block to switch clip. A new clip inherits
your references so you're not reloading the same five images every time.
`Save packed` writes a `.h3proj.zip` with the media inside — my whole
twelve-clip film is about 42 MB.

### 🔗 Links

The little 🔗 between two blocks on the timeline is the whole idea:

- **Linked** — the second clip *continues* the first. When it renders, it picks
  up the tail of the first one's render through CONTINUE FROM, motion and
  sound. One continuous take, longer than H3 gives you in one go.
- **Not linked** — it's a cut. The clip starts fresh.

Click the 🔗 to toggle it. There's no mode to pick anywhere else, the links
are the only place that decides.

### Render

At the bottom of the Project panel, pick what to render and hit **Render**:

- **this clip**
- **this chain** — the linked run this clip is part of
- **from this clip on**
- **whole project**

Linked clips continue from each other and Glide Join stitches as it goes, so
the **last file of a chain is the whole chain**. No separate assembly step.

**+ look** (on by default) — on every *unlinked* clip, hand it the clearest
window of the clip before as a video reference. Separate shots, same room,
same light. It takes video slot 1 on those clips.

While it runs the node is locked and shows which clip it's on. **Stop**
finishes the clip it's rendering and stops there — everything already done
stays done. If a render fails it stops, tells you which clip, and keeps the
finished ones.

If you don't like the result, **Revert** in the Project panel puts the project
back exactly as it was before the run.

### Skipping a clip

Each row in the project list has a dot. Click it and that clip drops out of the
run — the row dims, and Render passes over it. The clip stays in the project.
Click again to bring it back. For when one shot in the middle needs another
pass and you don't want to re-render the ten that already work.

---

## Latent hold — chains that don't cook

This one I'm proud of.

Chaining used to work like this: decode the previous clip to pixels, take its
last 22 frames, encode them again, hand them to H3. Every link did that round
trip, and every round trip cooked the picture a little more. I ran the same
jazz shot over six linked clips: by the end it was **14% less saturated** and
**55% crunchier** on the edges. Bricks with halos, haze turned blotchy.

My first idea was to pull the colour back toward the reference. Didn't help at
all. The problem wasn't the colour, it was the round trip.

So now H3 Studio continues from the previous clip's **own latent** — the real
thing the sampler made, never decoded and re-encoded. Same six links:
brightness, contrast, saturation and detail flat to the third decimal. There's
still a tiny bit of drift from the model itself, but the picture stays clean.

**Setup — one wire:**

```
sampler ──output (LATENT)──> Glide Video  (the new "latent" input)
```

That's it. Glide Video then saves a `.h3latent` file next to every video it
writes, and the next linked clip finds it by itself.

- **Latent hold is on by default.** The checkbox is on the CONTINUE FROM card.
- **No `.h3latent` next to the clip?** It quietly falls back to the old guide,
  nothing breaks. Re-render the first clip of a chain once with the wire
  plugged in and you're set.
- **Take the latent from the first sampler.** If you upscale and refine after,
  that second latent is a different size and won't be used.
- **Check it worked:** the console says `latent hold: 7 latent frames (22
  pictures) from ...`, or why it fell back.

**Cleaning up:** each `.h3latent` is about 30 MB, and you only need it to
continue *from* that clip. Once a chain is finished: ⚙ Settings → **Delete
latents…** It tells you how many and how big first, and it never touches the
videos.

---

## Song mode

One song for the whole project. Each clip locks its own window of it into the
audio half of the latent, so H3 animates against the **real** song — the
drummer hits on the beat — instead of making up its own music.

**Setup:**

1. Project panel → **Song** → **Add song…** (or drop an audio file on it).
2. `from` — where in the song the project starts. `hold` — how hard it's
   locked (1 = exactly).
3. Want the whole song covered? Pick a clip length and hit **Lay out clips**.
   It builds one linked chain long enough for the song.
4. Render **whole project**.

The window each clip gets follows the links — a linked clip gives its 22
overlap frames back, so the song lands on the same frames as the picture
across every join. **lock** off keeps the song in the project but renders
without it. **× Remove song** takes it out.

Wire `locked_audio` from H3 Studio into Glide Join: it splices the original
song samples back in, so the final file has the real track, not the VAE's copy
of it.

---

## Timeline audio

The three audio slots have two modes, switch at the top of the audio section:

- **Reference** — H3 hears them and you point at them with `@audio` in the
  prompt. The usual way: a voice, a sound to copy.
- **Timeline** — they're **placed in the clip**. Each one is held at the time
  you set, untouched, and H3 generates its own sound around it. A line of
  dialogue at 3 s, a door slam at the end, two songs with H3 singing the bit
  in between — I tried that last one, it transitions on its own. No `@audio`
  tokens in this mode, describe the sound in words.

The lane shows the clip's own time, with each audio as a coloured block:

- **Drop an audio file straight on the lane.** It goes into the first empty
  slot, starting where you let go.
- **Drag a block** to move it, **drag an edge** to trim it. Arrow keys nudge a
  frame, Shift a second.
- A long file is trimmed to fit the clip, so the edges move right away.
- `at` / `from` / `to` / `len` / `hold` on each card if you want exact numbers.

**▶ Play** on the left of the lane plays what this clip will hold: its part of
the song with the placed audio on top. **Click or drag the ruler** to move the
red playhead. When it reaches the end it goes back to where you pressed play,
so you can loop a section.

---

## Glide Audio Lock

The same audio lock as a standalone node, for your own workflows: it pins a
real track into the audio half of an H3 latent so H3 generates the picture
against it instead of re-performing it.

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
- `filename_prefix` — a slash makes a subfolder (`jazz/clip`), and date
  placeholders work: `%date:yyyy-MM-dd%`. Thanks Muxelmann for that one 👍
- `save_metadata` embeds the prompt and workflow in the file comment. Drag-drop
  restore into ComfyUI is reliable for MKV, hit or miss for MP4.
- `fallback_on_failure` retries with a more compatible preset instead of
  throwing away a finished render. It says so loudly when it does.
- `latent` (optional) — the first sampler's latent, saved next to the video as
  a `.h3latent` for latent hold.

The 4:4:4 and ProRes presets don't play in a browser, so the node writes a small
playable copy next to the real file just for the preview. The real file is the
one in your output folder — the copy is only so you can watch it in the node.

---

## Glide Join — extending a clip

H3 gives you a few seconds at a time. This is how you get more.

H3 Studio's **CONTINUE FROM** slot holds the tail of your previous clip, so the
model continues the motion and the sound instead of starting fresh. Glide Join
then assembles the result onto the source. In a project with links, Render
fills CONTINUE FROM for you.

**Wiring:**

```
H3 Studio ──positive──> sampler ──> VAE Decode ──images──> Glide Join ──> Glide Video
          ──overlap_frames──────────────────────────────>
          ──source_video────────────────────────────────>
          ──locked_audio────────────────────────────────>
                                    VAE Decode Audio ──audio──>
          sampler ──output (LATENT)───────────────────────────────────> Glide Video
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
- `match_levels` — fits gain and offset per frame against the source across the
  overlap. Leave it on. With latent hold it has almost nothing left to do,
  which is the point.
- `match_tail` — H3 rounds its audio grid up, so every clip carries about 8 ms
  more sound than picture. That grows at every join. Leave it on.

**Leave the CONTINUE FROM slot empty and the whole thing is a normal
generation** — H3 Studio sends nothing, Glide Join passes the frames straight
through. One workflow for both, no rewiring, no bypassing.

The window is **locked to the tail** of the clip, on purpose. A window from
the middle produces material that follows on from the middle, and splicing that
after the last frame skips everything between — it reads as a hard cut and no
seam mode can fix it.

### The chroma badge

This one matters when latent hold falls back to the guide (no `.h3latent`
next to the clip).

There's a little badge on the continuation waveform. Green for 4:4:4 and 4:2:2,
orange for 4:2:0. It's not decoration.

I had a continuation that kept coming back slightly out of sync — Glide Join
kept printing `frame correspondence was off by -2`. I went looking at the
anchor window, the frame counts, the way the tail is read. Wasted a good while
in there. It was the codec: AV1, which is 4:2:0. Same pair again as H.264
4:4:4 and the line just disappeared. ProRes 4:2:2, also clean.

So: **don't chain from a 4:2:0 file.** The badge tells you before you render.
A look reference carries no timing, so there's no badge on a reference video
slot.

---

## Careful with this

- **A `.py` change needs a full ComfyUI restart**, not a browser refresh.
- **If a node's widget list changes shape between versions, delete and re-add
  the node.** ComfyUI matches saved widget values by position and will smear old
  values across new widgets.
- **Long chains still drift a little**, even with latent hold — it's the model
  pushing its own look at every link. Fine for a few links; for a long one-take,
  grade it in post or cut and start a fresh chain.
- There will be bugs — open an issue and tell me what you did, that's genuinely
  more useful than a star.

---

## Credits

- MiniMax for H3, and the ComfyUI team for the core H3 nodes this builds on.
- **madebyollin** for the TAE decoder that makes the preview readable.
- **Kijai** for the quantised models that put this on smaller cards.
- **Muxelmann** for the date placeholders in Glide Video — first outside
  contribution to the pack 🙏

Development assistance from Claude (Anthropic).
