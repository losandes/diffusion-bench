"""
Upscaler registry. Backends are imported lazily so lightweight ones (lanczos)
work even when the heavier ML dependencies aren't importable.
"""
UPSCALERS = ["realesrgan", "lanczos"]


def get_upscaler(name, device="cpu", weights=None):
    if name == "lanczos":
        from .backends.lanczos import Lanczos

        return Lanczos()
    if name == "realesrgan":
        from .backends.realesrgan import RealESRGAN

        return RealESRGAN(device=device, weights=weights)
    raise Exception(f"Unknown upscaler: {name!r} (choices: {', '.join(UPSCALERS)})")
