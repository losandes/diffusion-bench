# Plan: standalone upscaling step (images + video)

A **separate** step from the generation pipeline (`python3 -m src.upscale`), for
upscaling both images and video. Starts with Real-ESRGAN; built so more backends
(video-native VSR, diffusion) drop in later. Not wired into txt2img/img2img/
vid2vid or the `-m` ensemble chaining.

## Why a separate step
The user wants upscaling decoupled from generation: generate at ~512px with
vid2vid, then upscale to the source resolution as its own command. This also
makes the upscaler reusable for arbitrary images/videos.

## Honest expectation
Upscaling from a 512px generation to (say) 1080x1920 **synthesizes plausible
detail** — it can't recover detail that was never generated. Choice of backend is
mostly about temporal stability for video.

## Architecture
```
src/upscale/
  __main__.py      # own argparse; detect image vs video by extension; dispatch
  driver.py        # streaming decode -> backend -> resize-to-target -> encode
  registry.py      # name -> backend (lazy imports so lanczos works without ML deps)
  backends/
    __init__.py    # Upscaler base (native_scale, upscale_frames)
    lanczos.py     # deterministic baseline (no model) — plumbing + fallback
    realesrgan.py  # Phase U1 — spandrel-loaded GAN, per-frame
    # realbasicvsr.py / basicvsrpp.py  (Phase U2 — video-native, temporal)
    # diffusion.py                      (Phase U3 — SD x4 / Stream-DiffVSR)
```

**Backend interface** (fits per-frame AND video-native):
```python
class Upscaler:
    native_scale = 1
    def upscale_frames(self, frames):  # iterable[PIL] -> iterator[PIL]
```
Real-ESRGAN maps per-frame; future video-native backends buffer temporal windows
(reuse pipelines/windowing.py). The **driver** is shared: it runs the backend,
then Lanczos-resizes to the exact target, and handles I/O identically for images
(a length-1 sequence) and video (a stream). Memory stays bounded (streaming).

## CLI
```shell
python3 -m src.upscale -i INPUT -o OUTDIR --upscaler realesrgan \
  --scale 2                 # multiplier, OR
  --to 1920x1080            # target as WIDTHxHEIGHT (aspect-preserved box)
  [--fps N] [--max_frames N]  # video only
  [--weights PATH]            # override Real-ESRGAN weights
```
- `--to` is **WIDTHxHEIGHT** (width first). Fit within the box, aspect preserved.
- target = `--to` or `--scale`; if neither, the backend's native scale.
- Output: image -> upscaled image; video -> upscaled `.mp4` + `.json` sidecar.

## Dependencies
`spandrel` (clean ESRGAN-family loader, MPS-capable; avoids the legacy `basicsr`
install). Real-ESRGAN weights auto-downloaded from the official GitHub release to
`~/.cache/diffusion-bench/weights` (override via `--weights` /
`DIFFUSION_BENCH_REALESRGAN_WEIGHTS`).

## Phases (each its own commit)
- **U1 (this one)** — entry point + shared driver + image/video I/O + Lanczos
  baseline + Real-ESRGAN. Lanczos makes plumbing testable headless; Real-ESRGAN
  validated on MPS (weights download).
- **U2** — video-native temporal backends (RealBasicVSR/BasicVSR++) via windowing.
- **U3** — diffusion upscalers (SD x4 / Stream-DiffVSR).

## Pairs with generation (two commands)
```shell
./run.sh -p "..." -m guoyww/animatediff-v1-5-2 -i clip.mp4 --fps 12
python3 -m src.upscale -i images/clip-animatediff-*.mp4 --to 1920x1080
```
Also `upscale.sh` mirrors `run.sh` (nice/threads) for background runs.
