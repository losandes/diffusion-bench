"""
Shared upscale driver: streaming decode -> backend -> resize-to-target -> encode.

The same code path serves images (a length-1 frame sequence) and video (a lazy
frame stream), so memory stays bounded regardless of clip length.
"""
import itertools
import os

from PIL import Image

from ..files.load import load_image
from ..files.video import VideoWriter, probe, read_frames, write_sidecar

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".gif", ".avi", ".m4v"}


def _even(n):
    """Round to a positive even int (many video codecs require even dimensions)."""
    n = int(round(n))
    return max(2, n - (n % 2))


def _target_size(src_w, src_h, to, scale, native_scale):
    """
    Computes the exact output size.

    to (h, w)?   - fit within this box, preserving aspect ratio (may upscale)
    scale (float)? - multiply source dimensions
    else         - use the backend's native scale factor
    """
    if to is not None:
        box_h, box_w = to
        factor = min(box_w / src_w, box_h / src_h)
    elif scale is not None:
        factor = scale
    else:
        factor = native_scale
    return _even(src_w * factor), _even(src_h * factor)


def _resized(frames, size):
    for frame in frames:
        yield frame if frame.size == size else frame.resize(size, Image.LANCZOS)


def _sidecar_meta(in_path, upscaler, target):
    return {
        "step": "upscale",
        "upscaler": upscaler,
        "source": in_path,
        "resolution": f"{target[0]}x{target[1]}",
    }


def upscale_image(in_path, out_path, backend, upscaler, to=None, scale=None):
    img = load_image(in_path)
    target = _target_size(img.width, img.height, to, scale, backend.native_scale)
    out = next(_resized(backend.upscale_frames([img]), target))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.save(out_path)
    write_sidecar(out_path, _sidecar_meta(in_path, upscaler, target))
    return out_path, target


def upscale_video(
    in_path, out_path, backend, upscaler, to=None, scale=None, fps=None, max_frames=None
):
    frames = read_frames(in_path, fps=fps, max_frames=max_frames)
    try:
        first = next(frames)
    except StopIteration:
        raise Exception(f"No frames decoded from {in_path}")

    target = _target_size(first.width, first.height, to, scale, backend.native_scale)
    print(f"upscale: {first.width}x{first.height} -> {target[0]}x{target[1]} ({upscaler})")

    source = itertools.chain([first], frames)
    upscaled = _resized(backend.upscale_frames(source), target)

    out_fps = fps or probe(in_path)["fps"] or 16
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    count = 0
    with VideoWriter(out_path, fps=out_fps) as writer:
        for frame in upscaled:
            writer.append(frame)
            count += 1

    write_sidecar(
        out_path,
        {**_sidecar_meta(in_path, upscaler, target), "out_fps": out_fps, "frames": count},
    )
    return out_path, count
