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

from PIL import Image

from ..constants import VID2VID
from ..files.video import VideoWriter, probe, read_frames, write_sidecar


def _swap_ext(path, ext):
    root, _ = os.path.splitext(path)
    return f"{root}.{ext.lstrip('.')}"


def _fit_source(options, src_w, src_h):
    """
    Fits a source frame (src_w x src_h) inside the -x/-y bounding box -- each side
    also bounded by the DIFFUSION_BENCH_MAX_SIZE cap (default 512) -- preserving
    the source aspect ratio, and rounds each dimension down to a multiple of 8
    (VAE requirement). Never upscales.

    -x/-y act as a *maximum* box for video (not an exact size), so the output
    keeps the clip's own aspect ratio regardless of orientation.

    Returns (width, height, cap, downscaled) where `downscaled` is True only when
    the box actually shrank the frame -- not when it merely rounded to /8.
    """
    cap = int(os.environ.get("DIFFUSION_BENCH_MAX_SIZE", "512"))
    box_w = min(options.get("width") or cap, cap)
    box_h = min(options.get("height") or cap, cap)
    scale = min(box_w / src_w, box_h / src_h, 1.0)  # preserve aspect, never upscale
    fit = lambda n: max(8, int(n * scale) // 8 * 8)
    return fit(src_w), fit(src_h), cap, scale < 1.0


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
    Temporally-coherent path (AnimateDiff), streamed in overlapping windows.

    The clip is decoded lazily and processed a window at a time
    (pipelines/windowing.py), so memory is bounded by one window regardless of
    clip length. Adjacent windows share --overlap frames that are crossfaded to
    hide seams, and every window uses the same seed so the style stays stable.

    -x/-y act as a maximum bounding box (source aspect is preserved); --window_size
    bounds memory per window.
    """
    import itertools

    import torch

    from ..pipelines.windowing import stitch

    model = options["model"]
    pipe = model["pipe"]

    # window_size / overlap: only fall back to defaults when actually unset, so a
    # legitimate --overlap 0 isn't swallowed by `0 or 4`.
    window_size = options.get("window_size")
    window_size = 16 if window_size is None else int(window_size)
    if window_size < 1:
        raise Exception("window_size must be at least 1")
    overlap = options.get("overlap")
    overlap = 4 if overlap is None else int(overlap)

    # Decode at native resolution and measure the FIRST decoded frame -- that
    # reflects any rotation the decoder applied, so aspect/orientation are correct
    # even for rotated phone clips. Fail early on an empty/undecodable clip.
    frame_iter = read_frames(
        in_path,
        fps=options.get("fps"),
        max_frames=options.get("max_frames"),
    )
    try:
        first = next(frame_iter)
    except StopIteration:
        raise Exception(f"No frames decoded from {in_path}")

    src_w, src_h = first.size
    # AnimateDiff is SD1.5-based; spatial self-attention scales with (H*W)^2, so
    # large frames blow up memory. Fit within the box, preserving aspect.
    width, height, cap, downscaled = _fit_source(options, src_w, src_h)
    if downscaled:
        print(
            f"AnimateDiff: processing at {width}x{height} (fit from source "
            f"{src_w}x{src_h}, max {cap}px; set DIFFUSION_BENCH_MAX_SIZE to change)"
        )

    # Resize the stream to the aspect-preserved target (decouples resize from decode).
    native_iter = itertools.chain([first], frame_iter)
    frame_iter = (frame.resize((width, height), Image.LANCZOS) for frame in native_iter)

    kwargs = {
        "num_inference_steps": options["steps"],
        "negative_prompt": options.get("negative_prompt"),
        "height": height,
        "width": width,
    }
    if options.get("strength") is not None:
        kwargs["strength"] = options["strength"]
    if options.get("guidance_scale") is not None:
        kwargs["guidance_scale"] = options["guidance_scale"]

    seeds = options.get("seeds") or []
    seed = int(seeds[0]) if seeds and seeds[0] is not None else None

    window_index = {"n": 0}

    def process_fn(win_frames):
        window_index["n"] += 1
        print(f"  window {window_index['n']}: {len(win_frames)} frames")
        call_kwargs = dict(kwargs)
        if seed is not None:
            # Re-seed each window identically so the added noise (and thus style)
            # is consistent across windows rather than drifting.
            call_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        return pipe(video=win_frames, prompt=options["prompt"], **call_kwargs).frames[0]

    out_fps = options.get("fps") or probe(in_path)["fps"] or 16

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with VideoWriter(out_path, fps=out_fps) as writer:
        count = stitch(process_fn, frame_iter, window_size, overlap, writer.append)

    write_sidecar(
        out_path,
        {
            "prompt": options["prompt"],
            "negative_prompt": options.get("negative_prompt"),
            "model": model["name"],
            "path": "coherent (animatediff, windowed)",
            "resolution": f"{width}x{height}",
            "steps": options["steps"],
            "strength": options.get("strength"),
            "guidance_scale": options.get("guidance_scale"),
            "window_size": window_size,
            "overlap": overlap,
            "windows": window_index["n"],
            "source": in_path,
            "out_fps": out_fps,
            "frames": count,
        },
    )

    return out_path, count


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

    print("")
    print(f"model_id: {options['model']['name']}")
    print(f"prompt:   {options['prompt']}")
    print(f"input:    {in_path}")
    print(f"output:   {out_path}")
    print("")

    # AnimateDiff (VID2VID) models use the coherent path unless --naive is set.
    if options["model"]["type"] == VID2VID and not options.get("naive"):
        return _coherent_run(options, in_path, out_path)

    # Naive path: fit the source within the -x/-y box, preserving aspect ratio
    # (measured from a decoded frame so rotated clips aren't distorted).
    peek = next(read_frames(in_path), None)
    if peek is None:
        raise Exception(f"No frames decoded from {in_path}")
    tw, th, cap, downscaled = _fit_source(options, *peek.size)
    if downscaled:
        print(f"vid2vid: processing at {tw}x{th} (fit from source {peek.size[0]}x{peek.size[1]}, max {cap}px)")
    size = (tw, th)

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
