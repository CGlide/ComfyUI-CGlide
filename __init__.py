from .csglide_vision import CSGlideVision
from .csglide_seed import CSGlideSeed
from .csglide_video import CSGlideVideo
from .csglide_cast import CSGlideCast
from .csglide_preview import CSGlidePreview
from .csglide_join import CSGlideJoin

NODE_CLASS_MAPPINGS = {
    "CSGlideVisionCS": CSGlideVision,
    "CSGlideSeedCS": CSGlideSeed,
    "CSGlideVideoCS": CSGlideVideo,
    "CSGlideCastCS": CSGlideCast,
    "CSGlidePreviewCS": CSGlidePreview,
    "CSGlideJoinCS": CSGlideJoin,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSGlideVisionCS": "Glide Vision",
    "CSGlideSeedCS": "Glide Seed",
    "CSGlideVideoCS": "Glide Video",
    "CSGlideCastCS": "H3 Studio",
    "CSGlidePreviewCS": "Glide Preview",
    "CSGlideJoinCS": "Glide Join",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
