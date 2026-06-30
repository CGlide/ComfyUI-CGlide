from .csglide_vision import CSGlideVision
from .csglide_seed import CSGlideSeed

NODE_CLASS_MAPPINGS = {
    "CSGlideVisionCS": CSGlideVision,
    "CSGlideSeedCS": CSGlideSeed,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CSGlideVisionCS": "Glide Vision",
    "CSGlideSeedCS": "Glide Seed",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
