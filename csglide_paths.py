"""Path containment for every file a CGlide node or route reads or writes.

Everything that names a file - a node widget, a route query, a value wired
from another node - can arrive from anyone who reaches ComfyUI's port: POST
/prompt has no login, so a widget value is as untrusted as a request body.
The rule here is the one ComfyUI's own loaders follow: files live inside
ComfyUI's input, output and temp folders, and nowhere else.

Two passes, in this order:

1. The TEXT is checked before anything touches the disk. A UNC path
   (\\\\host\\share) or one starting with // makes Windows open an SMB connection
   to the named host on the first stat(), so it has to be refused before the
   first filesystem call, not after. Same for '..' and NUL. An absolute path
   is refused unless it already spells out a place inside one of the roots:
   H3 Studio hands its own render paths to Glide Join and Glide Upscale Hold
   as full paths inside output/, and those keep working.
2. The surviving path is resolved (realpath, so symlinks count) and has to
   land inside one of the roots by os.path.commonpath.
"""
import os
import re

import folder_paths


class PathRefused(ValueError):
    """The path is outside ComfyUI's folders. Message is safe to show."""


_DRIVE = re.compile(r"^[A-Za-z]:")


def _root_dirs(kinds):
    out = []
    for kind in kinds:
        try:
            if kind == "input":
                d = folder_paths.get_input_directory()
            elif kind == "output":
                d = folder_paths.get_output_directory()
            elif kind == "temp":
                d = folder_paths.get_temp_directory()
            else:
                continue
        except Exception:
            continue
        if d:
            out.append(d)
    return out


ALL_ROOTS = ("input", "output", "temp")


def _norm(p):
    return os.path.normcase(os.path.normpath(p))


def _under(root, target):
    try:
        return os.path.commonpath([root, target]) == root
    except ValueError:            # different drives on Windows
        return False


def _textually_inside(path, roots):
    """No filesystem call: does the spelled-out path sit under a root?"""
    target = _norm(path)
    # Both spellings of each root: as configured, and resolved. An output
    # folder that is a symlink to another drive is common, and the paths
    # other nodes hand over are already resolved. Resolving a ROOT is fine:
    # it is the operator's own path, not the caller's.
    spellings = []
    for r in roots:
        spellings.append(_norm(os.path.abspath(r)))
        spellings.append(_norm(os.path.realpath(r)))
    return any(_under(r, target) for r in spellings)


def is_absolute_text(raw):
    return (raw.startswith(("/", "\\")) or bool(_DRIVE.match(raw))
            or os.path.isabs(raw))


def clean_text(raw):
    """Strip quotes and spaces, then refuse what must never reach the disk.
    Returns the cleaned string ('' for empty input). Touches nothing."""
    s = str(raw or "").strip().strip('"').strip("'").strip()
    if not s:
        return ""
    if "\x00" in s:
        raise PathRefused("That file name has a NUL character in it.")
    if s.startswith(("\\\\", "//", "\\/", "/\\")):
        raise PathRefused(
            "Network (UNC) paths are not accepted. Put the file in ComfyUI's "
            "input or output folder.")
    parts = re.split(r"[\\/]+", s)
    if ".." in parts:
        raise PathRefused("Paths with '..' in them are not accepted.")
    return s


def safe_path(raw, bases=("input",), roots=ALL_ROOTS, kind="file"):
    """Resolve a user-supplied path inside ComfyUI's folders.

    raw    - the value as it arrived (widget, query, wired string).
    bases  - where a relative name is looked up, in order.
    roots  - the folders the result may live in.
    kind   - "file", "dir", or None (no existence check, e.g. a sidecar
             that may be missing; the containment check still applies).

    Returns the real path, or None when the name is fine but nothing is
    there. Raises PathRefused when the path points outside the roots."""
    s = clean_text(raw)
    if not s:
        return None
    root_dirs = _root_dirs(roots)
    if not root_dirs:
        raise PathRefused("ComfyUI's folders are not available.")

    if is_absolute_text(s):
        if not _textually_inside(s, root_dirs):
            raise PathRefused(
                "Only files inside ComfyUI's input, output or temp folders are "
                "read. Put the file there, or use its name in input/.")
        candidates = [s]
    else:
        candidates = [os.path.join(b, s) for b in _root_dirs(bases)]

    real_roots = [_norm(os.path.realpath(r)) for r in root_dirs]
    refused = False
    for cand in candidates:
        real = os.path.realpath(cand)
        if not any(_under(r, _norm(real)) for r in real_roots):
            refused = True                     # symlink pointing out
            continue
        if kind == "file" and not os.path.isfile(real):
            continue
        if kind == "dir" and not os.path.isdir(real):
            continue
        return real
    if refused:
        raise PathRefused("That path leads outside ComfyUI's folders.")
    return None


def safe_subdir(raw, root="output"):
    """A folder to write into: a subfolder name under one root (output by
    default), or a full path that is already inside it. Created if needed.
    Empty -> the root itself."""
    base = _root_dirs((root,))
    if not base:
        raise PathRefused("ComfyUI's output folder is not available.")
    base = base[0]
    s = clean_text(raw)
    if not s:
        return os.path.realpath(base)
    if is_absolute_text(s):
        if not _textually_inside(s, [base]):
            raise PathRefused(
                "output_directory has to be a folder inside ComfyUI's output "
                "folder. Type a name like Upscaled for output/Upscaled.")
        cand = s
    else:
        cand = os.path.join(base, s)
    real_base = _norm(os.path.realpath(base))
    # Check before mkdir, then again after: an existing symlink in the chain
    # resolves now, a new folder cannot point anywhere.
    if not _under(real_base, _norm(os.path.realpath(cand))):
        raise PathRefused("output_directory leads outside ComfyUI's output folder.")
    os.makedirs(cand, exist_ok=True)
    real = os.path.realpath(cand)
    if not _under(real_base, _norm(real)):
        raise PathRefused("output_directory leads outside ComfyUI's output folder.")
    return real
