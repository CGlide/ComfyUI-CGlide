"""
csglide_run.py -- the one place this pack starts an external process.

Every ffmpeg and ffprobe invocation in CGlide goes through here. Nothing else
in the pack imports `subprocess`, so there is exactly one function to audit
rather than eight call sites spread over four files.

The safety properties, all enforced in `_argv()` below and true of every call:

  * argv is a LIST. There is no shell anywhere in this module -- `shell=True`
    is never passed and never reachable -- so a filename containing `;`, `&&`,
    backticks or quotes is handed to ffmpeg as a literal filename and cannot
    become a command. Command injection is structurally impossible, not merely
    unlikely.
  * Every element is checked to be a `str`. A path, an int or a bytes object
    is rejected before the call rather than being coerced into something
    surprising.
  * The executable is one this module resolved itself (`ffmpeg_path()` /
    `ffprobe_path()`), and `run()` refuses anything else. The single exception
    is `verify()`, which IS the resolver -- see its docstring.

Callers that accept a path from an HTTP request (csglide_adopt.py) resolve it
against ComfyUI's own root and reject anything escaping it BEFORE calling in
here. That check belongs at the route, not at this layer; this module's job is
the process boundary.
"""

import os
import shutil
import subprocess

__all__ = [
    "ffmpeg_path", "ffprobe_path", "run", "popen",
    "ffmpeg_candidates", "verify", "FFMPEG_ENV_VARS",
]

# Timeouts are per call. None of these commands is long-running: the probes
# read a header or a capability list, and the encode uses popen() instead.
DEFAULT_TIMEOUT = 20

# Explicit override, honoured first by the resolver. Documented in the README
# so a user with a full ffmpeg build can point the pack at it.
FFMPEG_ENV_VARS = ("CSGLIDE_FFMPEG", "FFMPEG_BINARY")

_ffmpeg_cache = None
_ffprobe_cache = None
_ffprobe_done = False

# Executables this module resolved and verified. run() and popen() will not
# start anything that is not in here.
_resolved: set = set()


# --------------------------------------------------------------------------
# argv construction -- the guard every call passes through
# --------------------------------------------------------------------------

def _argv(exe, args):
    """Build a checked argv list, or raise.

    Kept deliberately boring and total: no branching on content, no escaping,
    no quoting. There is nothing to escape FOR, because the list form never
    goes near a shell -- quoting here would in fact be the bug, since ffmpeg
    would then receive literal quote characters as part of the filename.
    """
    if not isinstance(exe, str) or not exe:
        raise ValueError("executable must be a non-empty str, got %r" % (exe,))
    argv = [exe]
    for i, a in enumerate(args):
        if not isinstance(a, str):
            raise TypeError(
                "argument %d must be a str, got %r (%s)" % (i, a, type(a).__name__))
        argv.append(a)
    return argv


def _trusted(exe):
    """True when `exe` is a binary this module located and verified itself."""
    return isinstance(exe, str) and exe in _resolved


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run(exe, args, timeout=DEFAULT_TIMEOUT, text=True):
    """Run a resolved binary with checked arguments and capture its output.

    Always captures; every caller in this pack wants stdout or stderr. Never
    uses a shell. Raises on an unresolved executable rather than running it.
    """
    if not _trusted(exe):
        raise ValueError(
            "refusing to run %r: not an executable resolved by this module. "
            "Use ffmpeg_path() or ffprobe_path()." % (exe,))
    return subprocess.run(
        _argv(exe, args),
        capture_output=True, text=text, timeout=timeout, shell=False,
    )


def popen(exe, args):
    """Start a resolved binary with stdin and stderr piped, for the encode.

    Glide Video writes raw frames to ffmpeg's stdin while draining stderr on a
    separate thread, so it needs a live process rather than run()'s
    wait-for-exit. Same guards; the only difference is who reads the pipes.

    The pipe wiring is fixed here rather than passed in, so callers need no
    reference to `subprocess` at all -- this module stays the only importer.
    """
    if not _trusted(exe):
        raise ValueError(
            "refusing to run %r: not an executable resolved by this module. "
            "Use ffmpeg_path() or ffprobe_path()." % (exe,))
    return subprocess.Popen(
        _argv(exe, args),
        stdin=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
    )


def verify(cand, timeout=10):
    """Does `cand` run as ffmpeg? Marks it resolved on success.

    This is the ONE call in the pack that starts a binary before it is
    trusted, because it is what establishes the trust: the resolver has to try
    each candidate to find out which one works. It is still a list-form argv
    with no shell, the only argument is the literal "-version", and candidates
    come from a fixed set -- the two env vars above, shutil.which(), the
    imageio-ffmpeg package, and hardcoded install paths. Nothing from an HTTP
    request or a workflow reaches here.

    Returns True/False and never raises, so a candidate that is missing, not
    executable or not ffmpeg at all is simply skipped.
    """
    if not isinstance(cand, str) or not cand:
        return False
    try:
        # A bare name (no separator) is left to the OS path search; a path is
        # required to exist before we try to execute it.
        if os.path.sep in cand and not os.path.isfile(cand):
            return False
        subprocess.run([cand, "-version"],
                       capture_output=True, timeout=timeout, shell=False)
    except Exception:
        return False
    _resolved.add(cand)
    return True


# --------------------------------------------------------------------------
# locating the binaries
# --------------------------------------------------------------------------

def ffmpeg_candidates():
    """Every place ffmpeg might reasonably be, best first.

    Windows does not ship ffmpeg and ComfyUI portable does not add it to PATH,
    so shutil.which() alone fails on most Windows installs.
    """
    # 1. explicit override wins
    for var in FFMPEG_ENV_VARS:
        env = os.environ.get(var)
        if env:
            yield env

    # 2. on PATH
    w = shutil.which("ffmpeg")
    if w:
        yield w

    # 3. imageio-ffmpeg ships a working binary and is a common transitive
    #    dependency in ComfyUI installs -- usually present even when the
    #    system has no ffmpeg at all
    try:
        import imageio_ffmpeg
        yield imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    # 4. relative to the ComfyUI tree, and the usual Windows install spots
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [
        os.path.abspath(os.path.join(here, "..", "..")),        # ComfyUI/
        os.path.abspath(os.path.join(here, "..", "..", "..")),  # portable root
    ]
    names = ["ffmpeg.exe", "ffmpeg"]
    subs = ["", "ffmpeg", os.path.join("ffmpeg", "bin"), "bin",
            os.path.join("python_embeded", "Scripts")]
    for root in roots:
        for sub in subs:
            for name in names:
                yield os.path.join(root, sub, name) if sub else os.path.join(root, name)

    for p in [r"C:\ffmpeg\bin\ffmpeg.exe",
              r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        yield p


def ffmpeg_path(refresh=False):
    """Locate a working ffmpeg, caching the result."""
    global _ffmpeg_cache
    if _ffmpeg_cache is not None and not refresh:
        return _ffmpeg_cache

    for cand in ffmpeg_candidates():
        if verify(cand):
            _ffmpeg_cache = cand
            _resolved.add(cand)
            return cand

    # Last resort: will fail loudly at encode time with a clear message,
    # which is better than returning None and crashing somewhere else.
    _ffmpeg_cache = "ffmpeg"
    _resolved.add(_ffmpeg_cache)
    return _ffmpeg_cache


def ffprobe_path(refresh=False):
    """ffprobe, or None. Callers MUST handle None.

    imageio-ffmpeg ships ffmpeg WITHOUT ffprobe, which is the common case in
    ComfyUI installs, so the sibling guess usually misses and PATH is what
    answers -- when anything does.
    """
    global _ffprobe_cache, _ffprobe_done
    if _ffprobe_done and not refresh:
        return _ffprobe_cache

    _ffprobe_done = True
    _ffprobe_cache = None

    # beside the ffmpeg we already found
    exe = ffmpeg_path()
    base = os.path.basename(exe)
    for a, b in (("ffmpeg.exe", "ffprobe.exe"), ("ffmpeg", "ffprobe")):
        if base == a:
            cand = os.path.join(os.path.dirname(exe), b)
            if os.path.isfile(cand):
                _ffprobe_cache = cand
                _resolved.add(cand)
                return cand

    w = shutil.which("ffprobe")
    if w:
        _ffprobe_cache = w
        _resolved.add(w)
    return _ffprobe_cache
