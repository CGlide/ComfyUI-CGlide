from .csglide_vision import CSGlideVision
from .csglide_seed import CSGlideSeed
from .csglide_video import CSGlideVideo
from .csglide_cast import CSGlideCast
from .csglide_preview import CSGlidePreview
from .csglide_join import CSGlideJoin
from .csglide_dlss5 import CSGlideDLSS5, CSGlideDLSS5Batch, CSGlideDLSS5Legacy

# Routes only, no node class: importing it registers /cglide/recent_outputs and
# /cglide/adopt_output on ComfyUI's server. Nothing to add to the mappings below.
from . import csglide_adopt  # noqa: F401

NODE_CLASS_MAPPINGS = {
    "CSGlideVisionCS": CSGlideVision,
    "CSGlideSeedCS": CSGlideSeed,
    "CSGlideVideoCS": CSGlideVideo,
    "CSGlideCastCS": CSGlideCast,
    "CSGlidePreviewCS": CSGlidePreview,
    "CSGlideJoinCS": CSGlideJoin,
    "CSGlideDLSS5CS": CSGlideDLSS5,
    "CSGlideDLSS5BatchCS": CSGlideDLSS5Batch,
    # The id the standalone ComfyUI-Glide-DLSS5 folder used, so workflows
    # saved with it still open. Hidden from search (DEPRECATED).
    "CSGlideDLSS5": CSGlideDLSS5Legacy,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSGlideVisionCS": "Glide Vision",
    "CSGlideSeedCS": "Glide Seed",
    "CSGlideVideoCS": "Glide Video",
    "CSGlideCastCS": "H3 Studio",
    "CSGlidePreviewCS": "Glide Preview",
    "CSGlideJoinCS": "Glide Join",
    "CSGlideDLSS5CS": "Glide DLSS5",
    "CSGlideDLSS5BatchCS": "Glide DLSS5 Batch",
    "CSGlideDLSS5": "Glide DLSS5 (old)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
