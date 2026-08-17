"""
csglide_ffmpeg_check.py -- diagnose why Glide Video only offers H.264.

Run with ComfyUI's own python so it sees the same packages ComfyUI does:

    S:\\ComfyUI\\python_embeded\\python.exe csglide_ffmpeg_check.py

It reports which ffmpeg was found and which encoders that build has.
"""

import os
import shutil
import subprocess

WANTED = {
    "libx264":   "H.264 (compatible)",
    "libx265":   "H.265 (smaller)",
    "av1_nvenc": "AV1 (small, best quality)  [GPU]",
    "libsvtav1": "AV1 fallback               [CPU]",
    "prores_ks": "ProRes 422 HQ (master)",
    "ffv1":      "FFV1 (lossless archive)",
}


def candidates():
    env = os.environ.get("CSGLIDE_FFMPEG") or os.environ.get("FFMPEG_BINARY")
    if env:
        yield "env var", env

    w = shutil.which("ffmpeg")
    if w:
        yield "PATH", w

    try:
        import imageio_ffmpeg
        yield "imageio-ffmpeg", imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        print("  imageio-ffmpeg not usable: %s" % e)

    for p in [r"C:\ffmpeg\bin\ffmpeg.exe",
              r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        if os.path.isfile(p):
            yield "common path", p


def encoders(exe):
    import re
    out = subprocess.run([exe, "-hide_banner", "-encoders"],
                         capture_output=True, text=True, timeout=20).stdout
    return {m.group(2) for m in re.finditer(r"^\s*([VAS][\.A-Z]{5})\s+(\S+)", out, re.M)
            if m.group(2) != "="}


def main():
    print("Looking for ffmpeg...\n")
    found = list(candidates())

    if not found:
        print("  NOTHING FOUND.\n")
        print("  Windows does not ship ffmpeg and ComfyUI portable does not")
        print("  add it to PATH, so this is the usual cause.\n")
        print("  Fix: download a full build (gyan.dev 'full' or BtbN), unzip")
        print("  to C:\\ffmpeg, then either add C:\\ffmpeg\\bin to PATH or set")
        print("  CSGLIDE_FFMPEG=C:\\ffmpeg\\bin\\ffmpeg.exe")
        return

    for source, exe in found:
        print("  [%s] %s" % (source, exe))
        try:
            enc = encoders(exe)
        except Exception as e:
            print("      cannot run: %s\n" % e)
            continue
        print("      %d encoders" % len(enc))
        for name, label in WANTED.items():
            print("      %-4s %-12s %s" % ("OK" if name in enc else "--", name, label))
        print()

    print("Glide Video uses the first entry above that works.")
    print("Only H.264 in the dropdown means either nothing was found, or the")
    print("build is minimal. A full build has all of the above except the")
    print("nvenc ones, which also need an NVIDIA GPU.")


if __name__ == "__main__":
    main()
