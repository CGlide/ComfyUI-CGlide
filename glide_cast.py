warning: in the working copy of 'csglide_cast.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/csglide_cast.py b/csglide_cast.py[m
[1mindex 452bda7..eb7c8d9 100644[m
[1m--- a/csglide_cast.py[m
[1m+++ b/csglide_cast.py[m
[36m@@ -493,9 +493,10 @@[m [mclass CSGlideCast:[m
             },[m
         }[m
 [m
[31m-    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT", "STRING")[m
[32m+[m[32m    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "INT", "INT", "FLOAT", "INT",[m
[32m+[m[32m                    "STRING", "IMAGE")[m
     RETURN_NAMES = ("positive", "latent", "width", "height", "length", "seconds",[m
[31m-                    "overlap_frames", "source_video")[m
[32m+[m[32m                    "overlap_frames", "source_video", "guide_frames")[m
     FUNCTION = "build"[m
     CATEGORY = "CGlide"[m
     DESCRIPTION = "MiniMax H3 director — first/last keyframes or omni references, with automatic reference tagging."[m
[36m@@ -601,13 +602,14 @@[m [mclass CSGlideCast:[m
         0 - rather than calling that node, so the tail can be taken from the[m
         right end and levelled before the encode.[m
 [m
[31m-        Returns (keyframe or None, frames anchored). The frame count is the[m
[32m+[m[32m        Returns (keyframe or None, frames anchored, the frames themselves).[m
[32m+[m[32m        The frame count is the[m
         head that comes back in the output and has to come off before the clip[m
         is joined to its predecessor.[m
         """[m
         cont = cfg.get("cont")[m
         if not cont:[m
[31m-            return None, 0[m
[32m+[m[32m            return None, 0, None[m
 [m
         want = snap_guide_run(cont["frames"])[m
         if want < MIN_REF_FRAMES:[m
[36m@@ -628,7 +630,14 @@[m [mclass CSGlideCast:[m
         frames = _load_video_tail(cont["file"], want + 17, cont["start"])[m
         if frames is None:[m
             frames = _load_video_frames(cont["file"], cont["start"], cont["end"], want + 17)[m
[31m-        n = min(snap_guide_run(frames.shape[0]), want)[m
[32m+[m[32m        # Snap LAST. It used to be min(snap(available), want), which returns[m
[32m+[m[32m        # `want` untouched whenever want is the smaller of the two - and want[m
[32m+[m[32m        # comes from the browser's window, which on mkv can round to 23 or 24[m
[32m+[m[32m        # where the file really holds 22. Anything that is not 17k+5 makes the[m
[32m+[m[32m        # model reserve rows for one more latent frame than the encode produces,[m
[32m+[m[32m        # which is the constant shape mismatch: same size every time for a given[m
[32m+[m[32m        # canvas, a different size for every canvas.[m
[32m+[m[32m        n = snap_guide_run(min(frames.shape[0], want))[m
         if n < MIN_REF_FRAMES:[m
             raise ValueError([m
                 "H3 Studio: %s gave only %d frames in that window; H3 needs %d. "[m
[36m@@ -639,6 +648,18 @@[m [mclass CSGlideCast:[m
         frames = _resize(frames, width, height, "center")[m
         keyframe = {"resolved_frame_index": 0, "latent": vae.encode(frames)}[m
 [m
[32m+[m[32m        # The shape mismatch reports live here. The model reserves rows from the[m
[32m+[m[32m        # SHAPE of this guide latent and then fills them from the same tensor, so[m
[32m+[m[32m        # if those two disagree the numbers below say by how much - and whether[m
[32m+[m[32m        # the guide is a legal 17k+5 run in the first place.[m
[32m+[m[32m        try:[m
[32m+[m[32m            print("[H3 Studio] continue: window=%d frames  canvas=%dx%d  "[m
[32m+[m[32m                  "guide_latent=%s  target_latent=%s  clip_frames=%d"[m
[32m+[m[32m                  % (n, width, height, tuple(keyframe["latent"].shape),[m
[32m+[m[32m                     tuple(latent["samples"].tensors[0].shape), frame_count))[m
[32m+[m[32m        except Exception as e:[m
[32m+[m[32m            print("[H3 Studio] continue: shape report unavailable (%s)" % e)[m
[32m+[m
         if cont["audio"]:[m
             # end=None: to EOF, so the carried sound finishes where the[m
             # picture does rather than at a duration the browser rounded[m
[36m@@ -658,7 +679,7 @@[m [mclass CSGlideCast:[m
         print("[H3 Studio] continue: %d frames from %s at frame 0%s, overlap %d"[m
               % (n, cont["file"],[m
                  " (levelled)" if cont["flatten"] > 0 else "", n))[m
[31m-        return keyframe, n[m
[32m+[m[32m        return keyframe, n, frames[m
 [m
     # ---------------- main ----------------[m
 [m
[36m@@ -667,8 +688,12 @@[m [mclass CSGlideCast:[m
 [m
         def finish(cond):[m
             """Shared tail for both modes: attach the continuation, if any."""[m
[31m-            guide, overlap = self._continuation([m
[32m+[m[32m            guide, overlap, guide_frames = self._continuation([m
                 cfg, vae, audio_vae, latent, width, height, frame_count)[m
[32m+[m[32m            # exactly what was encoded, so the anchored run can be LOOKED AT[m
[32m+[m[32m            # instead of inferred from the result of a six minute sample[m
[32m+[m[32m            if guide_frames is None:[m
[32m+[m[32m                guide_frames = torch.zeros((1, 64, 64, 3))[m
             if guide is not None:[m
                 keyframes = list(cond[0][1].get("minimax_keyframes", []))[m
                 keyframes.append(guide)[m
[36m@@ -677,7 +702,8 @@[m [mclass CSGlideCast:[m
             source = ""[m
             if cfg.get("cont"):[m
                 source = _resolve_asset(cfg["cont"]["file"]) or cfg["cont"]["file"][m
[31m-            return (cond, latent, width, height, frame_count, seconds, overlap, source)[m
[32m+[m[32m            return (cond, latent, width, height, frame_count, seconds, overlap,[m
[32m+[m[32m                    source, guide_frames)[m
 [m
         width = max(CANVAS_MULTIPLE, (cfg["width"] // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)[m
         height = max(CANVAS_MULTIPLE, (cfg["height"] // CANVAS_MULTIPLE) * CANVAS_MULTIPLE)[m
