"""
Real-ESRGAN backend, loaded via spandrel.

Per-frame GAN upscaler: sharp and deterministic (same input frame -> same output),
so on the already-coherent vid2vid output it adds little flicker. spandrel loads
the ESRGAN-family weights cleanly (no legacy basicsr install). Heavy deps (torch,
spandrel) are imported lazily so the lanczos backend works without them.
"""
import os

from . import Upscaler

# Official Real-ESRGAN x4 weights (photoreal). Override with --weights or
# DIFFUSION_BENCH_REALESRGAN_WEIGHTS (e.g. the anime variant).
_DEFAULT_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/"
    "RealESRGAN_x4plus.pth"
)
_CACHE_DIR = os.path.expanduser("~/.cache/diffusion-bench/weights")


def _download(url, dest):
    import requests

    print(f"Downloading Real-ESRGAN weights: {url}")
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        tmp = f"{dest}.part"
        with open(tmp, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
        os.replace(tmp, dest)


def _resolve_weights(weights):
    if weights:
        return weights
    env = os.environ.get("DIFFUSION_BENCH_REALESRGAN_WEIGHTS")
    if env:
        return env
    os.makedirs(_CACHE_DIR, exist_ok=True)
    dest = os.path.join(_CACHE_DIR, "RealESRGAN_x4plus.pth")
    if not os.path.exists(dest):
        _download(_DEFAULT_URL, dest)
    return dest


class RealESRGAN(Upscaler):
    def __init__(self, device="cpu", weights=None):
        import torch  # noqa: F401  (imported for side effects / availability)
        from spandrel import ModelLoader

        self.device = device
        path = _resolve_weights(weights)
        self.model = ModelLoader().load_from_file(path).to(device).eval()
        self.native_scale = getattr(self.model, "scale", 4)

    def upscale_frames(self, frames):
        import numpy as np
        import torch
        from PIL import Image

        for img in frames:
            arr = np.asarray(img.convert("RGB"))
            tensor = (
                torch.from_numpy(arr)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .unsqueeze(0)
                .to(self.device)
            )
            with torch.inference_mode():
                out = self.model(tensor)
            out = (
                out.squeeze(0)
                .clamp(0.0, 1.0)
                .mul(255.0)
                .round()
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            yield Image.fromarray(out)
