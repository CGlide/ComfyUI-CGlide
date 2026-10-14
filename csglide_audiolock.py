"""Glide Audio Lock (CSGlideAudioLockCS) — keep a real soundtrack intact through a MiniMax H3 render.

Drop between the AV latent (H3 Studio / EmptyMiniMaxH3LatentAV) and the sampler's
latent_image. The song is encoded with the H3 audio VAE into the audio half of the
packed latent, and a nested noise_mask pins those rows clean: ComfyUI's MiniMaxH3
model runs masked audio rows at AUDIO_COND_TIMESTEP and re-injects them every step,
so H3 generates the picture around the real track instead of re-performing it.

Outputs the locked latent plus the exact original audio segment, trimmed to the clip,
for muxing in Glide Video (bit-exact, no VAE round trip).
"""

import torch
import comfy.nested_tensor

AUDIO_LATENT_FPS = 40  # H3 audio latent rate


def _resample(waveform, sr, target_sr):
    if sr == target_sr:
        return waveform
    try:
        import comfy.audio
        return comfy.audio.resample(waveform, sr, target_sr)
    except Exception:
        import torchaudio
        return torchaudio.functional.resample(waveform, sr, target_sr)


def _encode(audio_vae, waveform, sr):
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    waveform = _resample(waveform, sr, vae_sr)
    return audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]


class CSGlideAudioLock:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "audio": ("AUDIO",),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3600.0, "step": 0.01,
                                            "tooltip": "Where in the song the clip starts."}),
                "lock_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                            "tooltip": "1.0 = audio kept exactly. Lower lets H3 rework it a little."}),
            }
        }

    RETURN_TYPES = ("LATENT", "AUDIO")
    RETURN_NAMES = ("latent", "audio")
    FUNCTION = "lock"
    CATEGORY = "CGlide"

    def lock(self, latent, audio_vae, audio, start_seconds, lock_strength):
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.unbind()) != 2:
            raise ValueError("Glide Audio Lock needs an H3 audio+video latent (video, audio pair).")
        video, target_audio = samples.unbind()
        audio_t = target_audio.shape[-1]
        clip_seconds = audio_t / AUDIO_LATENT_FPS

        # trim the song to the clip window
        waveform = audio["waveform"]
        sr = int(audio["sample_rate"])
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)  # H3 audio is stereo
        a = int(round(start_seconds * sr))
        b = a + int(round(clip_seconds * sr))
        segment = waveform[..., a:b]
        if segment.shape[-1] == 0:
            raise ValueError(f"start_seconds {start_seconds:.2f} is past the end of the audio.")

        z = _encode(audio_vae, segment, sr).to(device=target_audio.device, dtype=target_audio.dtype)
        rt = min(z.shape[-1], audio_t)

        new_audio = target_audio.clone()
        new_audio[..., :rt] = z[..., :rt].expand(new_audio.shape[0], -1, -1, -1)

        # mask: 0 = keep (pinned clean), 1 = generate. Any tail the song can't fill stays generated.
        audio_mask = torch.ones_like(new_audio)
        audio_mask[..., :rt] = 1.0 - float(lock_strength)

        # keep any video mask already on the latent (keyframes, continuation)
        video_mask = None
        old = latent.get("noise_mask", None)
        if old is not None:
            video_mask = old.unbind()[0] if getattr(old, "is_nested", False) else old
            if video_mask.shape != video.shape:
                video_mask = None  # foreign shape: let the sampler treat video as fully generated
        if video_mask is None:
            video_mask = torch.ones_like(video)

        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor((video, new_audio))
        out["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))

        # bit-exact original for muxing, padded with silence if the song ends early
        exact = segment
        need = int(round(clip_seconds * sr))
        if exact.shape[-1] < need:
            exact = torch.nn.functional.pad(exact, (0, need - exact.shape[-1]))
        print(f"[Glide Audio Lock] locked {rt}/{audio_t} audio latent frames "
              f"({rt / AUDIO_LATENT_FPS:.2f}s of {clip_seconds:.2f}s) from {start_seconds:.2f}s, strength {lock_strength}")
        return (out, {"waveform": exact[:1].contiguous(), "sample_rate": sr})
