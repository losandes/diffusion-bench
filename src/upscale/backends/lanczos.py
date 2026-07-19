from . import Upscaler


class Lanczos(Upscaler):
    """
    Deterministic baseline: no model. Passes frames through untouched; the driver
    performs the actual Lanczos resize to the target size. Temporally perfect
    (no added flicker) but synthesizes no new detail -- a free fallback and the
    reference the ML backends improve on.
    """

    native_scale = 1

    def upscale_frames(self, frames):
        return iter(frames)
