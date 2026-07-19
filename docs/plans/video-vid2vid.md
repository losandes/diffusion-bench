# Plan: prompt-driven mp4 transformation (video-to-video)

Transform an existing `.mp4` with a text prompt. Target model:
**AnimateDiff vid2vid** (temporally coherent) with a **naive per-frame img2img**
fallback for cheap plumbing/debug. Designed for **arbitrary-length** video via
streaming decode/encode and windowed processing.

## Key reframe

No diffusion pipeline consumes an mp4 container directly. Every approach
decodes the mp4 to frames (PIL/tensors), runs the model, and re-encodes.
`diffusers` ships `load_video()` / `export_to_video()` (imageio/pyav under the
hood, ffmpeg as encoder backend), so we don't hand-roll ffmpeg calls. The real
decision is *which model* transforms the frames:

| Strategy | Temporal quality | Cost | Verdict |
|---|---|---|---|
| Naive per-frame img2img | flickers (independent frames) | low | debug/fallback |
| AnimateDiff vid2vid (+ optional ControlNet) | coherent (cross-frame attn) | ~12GB, 512px, short windows | **primary** |
| Native video-latent (CogVideoX/Wan/SVD) | best, but txt2vid/img2vid | very high (~30-frame cap on M4/64GB) | out of scope |

## Design principles

1. **Streaming I/O** — never hold a whole video in RAM. Reader yields frames;
   we buffer one processing window at a time; writer appends output frames.
   Memory stays bounded regardless of clip length.
2. **Reuse the factory/dispatch pattern** — video is a new `type`, not a rewrite.
3. **Structure anchoring is the #1 quality lever** — optional ControlNet
   (lineart/depth via the existing `controlnet-aux` dep) locks each frame to the
   source geometry and does more for temporal stability than anything else.

## Phases

### Phase 0 — deps & scaffolding
- Add `imageio[ffmpeg]` to `pyproject.toml` (decode/encode + bundled ffmpeg).
- `src/constants.py` — add `VID2VID` type.
- `src/__main__.py` — dispatch branch `type == VID2VID -> vid2vid(...)`.

### Phase 1 — frame I/O (`src/files/video.py`)
- `read_frames(path, fps=None, max_frames=None, size=None)` — streaming
  generator of RGB PIL frames; optional fps subsample + resize to model res.
- `VideoWriter(path, fps)` — incremental `append(frame)`; probe source fps.
- `write_sidecar(path, meta)` — JSON sidecar (prompt/model/seed/steps/strength/
  fps/source) since EXIF is image-only and won't attach to an mp4 container.

### Phase 2 — pipeline registration (`src/pipelines/animatediff_vid2vid.py`)
- Factory returns `[model_id, pipe]`: `MotionAdapter` + SD1.5 base
  (default: Realistic Vision), DDIM/LCM scheduler, `enable_vae_slicing()` +
  `enable_attention_slicing()` for MPS, `enable_free_noise(...)` for long-range
  coherence.
- Register in `make_pipelines.py` with `type=VID2VID`, an **`in_channels`
  override** (video pipes have no `.unet`; the `pipe.unet.config.in_channels`
  read at make_pipelines.py:157 would crash), `supports_latents=False`.

### Phase 3 — windowing engine (`src/pipelines/windowing.py`) — DONE
Overlapping-window streaming: `stitch()` slides a fixed window across the clip
(bounded memory), crossfades `overlap` frames between windows, and re-seeds each
window identically for style stability. `_coherent_run` streams frames through it
to a `VideoWriter`. Constraint: `overlap <= window/2`. FreeNoise dropped in
favor of explicit windowing. (Original design notes below.)

- **Intra-window coherence:** AnimateDiff `FreeNoise` (noise reuse across a
  sliding context) — primary anti-flicker mechanism.
- **Inter-window memory bound + seams:** overlapping windows (e.g. 16 frames,
  4-frame overlap), linear crossfade blend on the overlap, fixed seed + carried
  tail latents across windows so style doesn't drift. Streaming: window N is
  written before window N+1 is decoded.
- Knobs: `window_size`, `overlap`, `stride`.

### Phase 4 — app module (`src/app/vid2vid.py`)
- Parallels `img2img.py:_refine` but loops over **windows of frames**, not
  `--count`. Per window: `pipe(video=frames, prompt, strength, guidance_scale,
  num_inference_steps, conditioning_frames=... if controlnet)` -> blend -> write.
- **Naive fallback:** loop the existing single-image refiner over frames.
- Optional ControlNet sub-mode: `controlnet-aux` (lineart/depth) per frame ->
  `conditioning_frames`.

### Phase 5 — CLI & templates (`src/options/get_args.py`, `make_paths.py`)
- New flags: `--strength`, `--guidance_scale`, `--fps`, `--max_frames`,
  `--window_size`, `--overlap`, `--controlnet` (off/lineart/depth), `--naive`.
  (`strength`/`guidance_scale` are currently never passed to any pipeline.)
- Relax the hardcoded `.png` default template so video passes emit `.mp4`.

### Phase 6 — validation (tiered, since GPU-dependent steps can't run headless)
1. frame I/O round-trip on a synthetic clip (headless).
2. naive path on a 1s clip (plumbing).
3. AnimateDiff single-window (~16 frames).
4. two-window blend (~32 frames) — seam check.

## Build order
Phase 0 -> 1 -> naive path (proves decode/encode/dispatch) -> 2 -> 3 -> 4
(AnimateDiff) -> ControlNet option -> windowing polish.

## Constraints to remember
- AnimateDiff is SD1.5-based: ~512px; the 896x640 default must downscale.
- Temporal models process a window at once -> RAM scales with
  `num_frames x resolution`; arbitrary length requires windowing (Phase 3).
- Local coherent v2v is practical for short windows; long video = many windows.
