class CSGlideSeed:
    """A simple seed source. Exposes the random/fixed/increment/decrement
    'control_after_generate' selector as a first-class widget, so it stays
    visible and usable even when promoted onto a group / subgraph node.
    Wire the INT output into a sampler's 'seed' input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "get_seed"
    CATEGORY = "CSGlide"

    def get_seed(self, seed):
        return (int(seed),)


NODE_CLASS_MAPPINGS = {"CSGlideSeedCS": CSGlideSeed}
NODE_DISPLAY_NAME_MAPPINGS = {"CSGlideSeedCS": "CSGlide Seed CS"}
