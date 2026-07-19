"""
Upscaler backends.

A backend enhances resolution/detail; the driver (src/upscale/driver.py) does the
final resize to the exact target and all I/O. Keeping backends I/O-free lets one
driver serve both images (a length-1 sequence) and video (a stream), and lets
future video-native backends buffer temporal windows behind the same interface.
"""


class Upscaler:
    """
    Base upscaler interface.

    native_scale: the factor the backend upscales by on its own (the driver
    resizes the result to the exact requested target afterward).
    """

    native_scale = 1

    def upscale_frames(self, frames):
        """iterable[PIL.Image] -> iterator[PIL.Image]"""
        raise NotImplementedError
