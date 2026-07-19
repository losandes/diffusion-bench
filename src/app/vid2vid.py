"""
Prompt-driven video-to-video.

This module owns the streaming decode -> transform -> encode loop. The core
(`transform_video`) is model-agnostic: it takes a `transform` callable applied
per frame, so the plumbing can be exercised with a passthrough/identity
transform (no model, no GPU) before AnimateDiff is wired in.

Phase 4 will add the temporally-coherent AnimateDiff path (windowed, via
pipelines/windowing.py). This file currently implements the NAIVE fallback:
each frame is refined independently, which is cheap but flickers.
"""
import os

from ..constants import VID2VID
from ..files.video import VideoWriter, probe, read_frames, write_sidecar


def _swap_ext(path, ext):
    root, _ = os.path.splitext(path)
    return f"{root}.{ext.lstrip('.')}"


def _fit_size(width, height, cap):
    """
    Scales (width, height) so the longest side is <= cap, preserving aspect
    ratio, and rounds each dimension down to a multiple of 8 (VAE requirement).
    Never upscales. Returns (width, height), each >= 8.
    """
    scale = min(1.0, cap / max(width, height))
    fit = lambda n: max(8, int(n * scale) // 8 * 8)
    return fit(width), fit(height)


def _derive_output_path(options):
    """
    Picks an .mp4 output path. The image pipeline templates output names as
    .png (see options/make_paths.py); until Phase 5 relaxes that, we reuse the
    templated path's directory + stem and force an .mp4 extension.
    """
    out_paths = options.get("output_paths") or []
    if out_paths and out_paths[0]:
        return _swap_ext(out_paths[0], "mp4")

    in_path = options["input_paths"][0]
    stem = os.path.splitext(os.path.basename(in_path))[0]
    out_dir = options.get("output_path") or "images"
    short = options.get("model", {}).get("short_name", "vid2vid")
    return os.path.join(out_dir, f"{stem}-{short}.mp4")


def transform_video(
    in_path,
    out_path,
    transform=None,
    *,
    fps=None,
    max_frames=None,
    size=None,
    sidecar_meta=None,
):
    """
    Streams frames from in_path, applies `transform` to each, and writes out_path.

    Parameters:
      in_path (str)          - source video
      out_path (str)         - destination .mp4
      transform (callable?)  - frame (PIL.Image) -> frame (PIL.Image); when None,
                               frames pass through unchanged (plumbing/passthrough)
      fps (float?)           - target output fps; defaults to the source fps
      max_frames (int?)      - cap on frames processed
      size ((w, h)?)         - resize frames before transforming
      sidecar_meta (dict?)   - provenance written to <out_path>.json

    Returns: (out_path, frame_count)
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    source = probe(in_path)
    out_fps = fps or source["fps"] or 16

    count = 0
    with VideoWriter(out_path, fps=out_fps) as writer:
        for frame in read_frames(in_path, fps=fps, max_frames=max_frames, size=size):
            writer.append(transform(frame) if transform is not None else frame)
            count += 1

    if sidecar_meta is not None:
        write_sidecar(out_path, {**sidecar_meta, "out_fps": out_fps, "frames": count})

    return out_path, count


def _coherent_run(options, in_path, out_path):
    """
    Temporally-coherent path (AnimateDiff). Reads the clip as a single window of
    frames and transforms them in one pipeline call, so the MotionAdapter's
    temporal layers keep the output consistent frame-to-frame.

    This is single-window: the whole (bounded) clip must fit in memory. Arbitrary
    length via overlapping windows + blend is Phase 3 (pipelines/windowing.py);
    until then, bound the clip with --max_frames and downscale with -x/-y.
    """
    model = options["model"]
    pipe = model["pipe"]

    # AnimateDiff is SD1.5-based and its spatial self-attention cost scales with
    # (H*W)^2, so large frames blow up memory (e.g. 1280x1920 needs a ~1.4TB
    # attention buffer). Cap the longest side and preserve aspect ratio; the cap
    # is overridable for machines with more headroom.
    cap = int(os.environ.get("DIFFUSION_BENCH_MAX_SIZE", "512"))
    req_w = options.get("width") or cap
    req_h = options.get("height") or cap
    width, height = _fit_size(req_w, req_h, cap)
    if (width, height) != (req_w, req_h):
        print(
            f"AnimateDiff: downscaling {req_w}x{req_h} -> {width}x{height} "
            f"(SD1.5/MPS memory cap {cap}px; set DIFFUSION_BENCH_MAX_SIZE to change)"
        )
    size = (width, height)

    frames = list(
        read_frames(
            in_path,
            fps=options.get("fps"),
            max_frames=options.get("max_frames"),
            size=size,
        )
    )
    if not frames:
        raise Exception(f"No frames decoded from {in_path}")

    window_size = options.get("window_size") or 16
    overlap = options.get("overlap") or 4

    # FreeNoise gives long-range coherence when the clip exceeds one context
    # window; for short clips a single context covers everything.
    if len(frames) > window_size and hasattr(pipe, "enable_free_noise"):
        pipe.enable_free_noise(context_length=window_size, context_stride=overlap)

    kwargs = {
        "num_inference_steps": options["steps"],
        "negative_prompt": options.get("negative_prompt"),
    }
    if options.get("strength") is not None:
        kwargs["strength"] = options["strength"]
    if options.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = options["guidance_scale"]
    kwargs["height"] = height
    kwargs["width"] = width

    result = pipe(video=frames, prompt=options["prompt"], **kwargs)
    out_frames = result.frames[0]

    source = probe(in_path)
    out_fps = options.get("fps") or source["fps"] or 16

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with VideoWriter(out_path, fps=out_fps) as writer:
        for frame in out_frames:
            writer.append(frame)

    write_sidecar(
        out_path,
        {
            "prompt": options["prompt"],
            "negative_prompt": options.get("negative_prompt"),
            "model": model["name"],
            "path": "coherent (animatediff, single-window)",
            "steps": options["steps"],
            "strength": options.get("strength"),
            "guidance_scale": options.get("guidance_scale"),
            "window_size": window_size,
            "source": in_path,
            "out_fps": out_fps,
            "frames": len(out_frames),
        },
    )

    return out_path, len(out_frames)


def _naive_frame_transform(options):
    """
    Builds a per-frame img2img transform around the pass's pipeline. Each frame
    is refined independently -- no temporal coherence (expect flicker). This is
    the debug/fallback path; the coherent path is AnimateDiff (Phase 4).
    """
    model = options["model"]
    prompt = options["prompt"]
    kwargs = {
        "num_inference_steps": options["steps"],
        "negative_prompt": options.get("negative_prompt"),
    }
    # strength / guidance_scale are Phase-5 CLI flags; pass them through when set.
    if options.get("strength") is not None:
        kwargs["strength"] = options["strength"]
    if options.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = options["guidance_scale"]

    def transform(frame):
        return model["pipe"](prompt, image=frame, **kwargs).images[0]

    return transform


def vid2vid(options, ensembleIdx=0):
    """
    Transforms an input video with a text prompt (naive per-frame path).

    Parameters:
      options (dict) - a pass produced by options/get_args.py
      ensembleIdx (int) - index of this step within an ensemble

    Returns: (out_path, frame_count)
    """
    in_paths = options.get("input_paths") or []
    if not in_paths:
        raise Exception("vid2vid requires an input video (-i / --input_paths)")

    in_path = in_paths[0]
    out_path = _derive_output_path(options)

    width = options.get("width")
    height = options.get("height")
    size = (width, height) if width and height else None

    print("")
    print(f"model_id: {options['model']['name']}")
    print(f"prompt:   {options['prompt']}")
    print(f"input:    {in_path}")
    print(f"output:   {out_path}")
    print("")

    # AnimateDiff (VID2VID) models use the coherent path unless --naive is set.
    if options["model"]["type"] == VID2VID and not options.get("naive"):
        return _coherent_run(options, in_path, out_path)

    transform = _naive_frame_transform(options)

    return transform_video(
        in_path,
        out_path,
        transform,
        fps=options.get("fps"),
        max_frames=options.get("max_frames"),
        size=size,
        sidecar_meta={
            "prompt": options["prompt"],
            "negative_prompt": options.get("negative_prompt"),
            "model": options["model"]["name"],
            "steps": options["steps"],
            "strength": options.get("strength"),
            "guidance_scale": options.get("guidance_scale"),
            "source": in_path,
            "ensemble_index": ensembleIdx,
        },
    )
