"""
csglide_ffmpeg_check.py -- diagnose why Glide Video only offers H.264.

Run with ComfyUI's own python so it sees the same packages ComfyUI does:

    S:\\ComfyUI\\python_embeded\\python.exe csglide_ffmpeg_check.py

It reports which ffmpeg was found and which encoders that build has.
"""

import os

# This script is run directly with ComfyUI's python, so it cannot use a
# package-relative import. csglide_run sits next to it either way.
sys_path_added = os.path.dirname(os.path.abspath(__file__))
if sys_path_added not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path_added)

import csglide_run as _run

WANTED = {
    "libx264":   "H.264 (compatible)",
    "libx265":   "H.265 (smaller)",
    "av1_nvenc": "AV1 (small, best quality)  [GPU]",
    "libsvtav1": "AV1 fallback               [CPU]",
    "prores_ks": "ProRes 422 HQ (master)",
    "ffv1":      "FFV1 (lossless archive)",
}


def candidates():
    """Every place the pack itself would look, in the same order.

    Shares csglide_run.ffmpeg_candidates() so this diagnostic cannot drift
    from what Glide Video actually does at encode time -- the whole point of
    the script is to explain that behaviour.
    """
    for cand in _run.ffmpeg_candidates():
        if cand and (os.path.sep not in cand or os.path.isfile(cand)):
            yield cand


def encoders(exe):
    import re
    # verify() is what promotes a candidate to runnable; the resolver
    # uses the same call, so this script tests exactly what ships.
    if not _run.verify(exe):
        raise RuntimeError("does not run as ffmpeg")
    out = _run.run(exe, ["-hide_banner", "-encoders"], timeout=20).stdout
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

    for exe in found:
        print("  %s" % exe)
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
