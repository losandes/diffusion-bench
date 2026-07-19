"""
Standalone upscaling entry point: `python3 -m src.upscale`.

Upscales images or video (detected by extension). Separate from the generation
pipeline. See docs/plans/upscale.md.
"""
import argparse
import os

from ..options.get_device_type import get_device_type
from ..options.split_to_list import split_to_list
from .driver import IMAGE_EXTS, VIDEO_EXTS, upscale_image, upscale_video
from .registry import UPSCALERS, get_upscaler


def _parse_to(value):
    """Parse a --to spec 'WIDTHxHEIGHT' (width first) into (width, height)."""
    if value is None:
        return None
    parts = value.lower().split("x")
    if len(parts) != 2 or not all(p.strip().isdigit() for p in parts):
        raise Exception(f"--to must be WIDTHxHEIGHT (e.g. 1920x1080), got {value!r}")
    return int(parts[0]), int(parts[1])


def _out_path(in_path, out_dir, upscaler):
    stem, ext = os.path.splitext(os.path.basename(in_path))
    ext = ".mp4" if ext.lower() in VIDEO_EXTS else ext
    return os.path.join(out_dir, f"{stem}-{upscaler}{ext}")


def main():
    parser = argparse.ArgumentParser(prog="python3 -m src.upscale")
    parser.add_argument("--input_paths", "-i", required=True, help="Comma-separated image/video path(s) to upscale")
    parser.add_argument("--output_path", "-o", default="images", help="Directory to write results (default=images)")
    parser.add_argument("--upscaler", "-u", default="realesrgan", choices=UPSCALERS, help="Upscaler backend (default=realesrgan)")
    parser.add_argument("--scale", type=float, help="Upscale multiplier (e.g. 2). Ignored if --to is given")
    parser.add_argument("--to", help="Target size as WIDTHxHEIGHT (e.g. 1920x1080), aspect-preserved")
    parser.add_argument("--weights", help="Path to model weights (overrides the backend default / auto-download)")
    parser.add_argument("--fps", type=float, help="Target output fps for video (default: source fps)")
    parser.add_argument("--max_frames", type=int, help="Cap on frames processed from a video")
    parser.add_argument("--device_type", "-d", help="Device (default: cuda/mps/cpu auto)")
    args = parser.parse_args()

    to = _parse_to(args.to)
    device = args.device_type or get_device_type()
    inputs = split_to_list(",")(args.input_paths)
    if not inputs:
        raise Exception("At least one input (-i / --input_paths) is required")

    backend = get_upscaler(args.upscaler, device=device, weights=args.weights)

    for in_path in inputs:
        ext = os.path.splitext(in_path)[1].lower()
        out_path = _out_path(in_path, args.output_path, args.upscaler)
        print(f"\ninput:    {in_path}\noutput:   {out_path}\n")

        if ext in VIDEO_EXTS:
            upscale_video(
                in_path, out_path, backend, args.upscaler,
                to=to, scale=args.scale, fps=args.fps, max_frames=args.max_frames,
            )
        elif ext in IMAGE_EXTS:
            upscale_image(in_path, out_path, backend, args.upscaler, to=to, scale=args.scale)
        else:
            raise Exception(
                f"Unsupported input extension {ext!r} for {in_path} "
                f"(images: {sorted(IMAGE_EXTS)}; video: {sorted(VIDEO_EXTS)})"
            )


if __name__ == "__main__":
    main()
