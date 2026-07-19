"""
Overlapping-window stitching for arbitrary-length video-to-video.

AnimateDiff denoises a whole window of frames in one pass, so memory scales with
the frame count -- a long clip in a single window OOMs (a 17s clip at 12fps is
~200 frames -> a ~46GB attention buffer). Instead we slide a fixed-size window
across the clip with a few overlapping frames, process each window
independently, and crossfade the overlap so seams don't pop.

Memory is bounded by one window (in + out) plus the overlap tail, regardless of
clip length. This module is model-agnostic: `stitch` takes a `process_fn` that
maps a list of frames to a list of frames, so it's testable with an identity
transform (no model, no GPU).
"""
import numpy as np
from PIL import Image


def iter_windows(frame_iter, window, stride):
    """
    Streams `window`-sized lists of frames from `frame_iter`, advancing by
    `stride` (so `window - stride` frames overlap between consecutive windows).

    The final short window is right-padded by repeating its last frame so every
    yielded window has exactly `window` frames; the pad count is returned so the
    caller can trim it back off. A window is only emitted if it introduces frames
    not already covered by the previous window.

    Yields: (frames_list, pad_count)
    """
    overlap = window - stride
    buf = []
    started = False

    for frame in frame_iter:
        buf.append(frame)
        if len(buf) == window:
            yield list(buf), 0
            started = True
            del buf[:stride]  # keep the trailing `overlap` frames

    if not started:
        # clip shorter than one window: a single (unpadded) window
        if buf:
            yield list(buf), 0
        return

    # buf now holds the carried `overlap` frames plus any new trailing frames
    new = len(buf) - overlap
    if new > 0:
        pad = window - len(buf)
        yield list(buf) + [buf[-1]] * pad, pad


def _blend(a, b, alpha):
    """Linear crossfade between two RGB PIL frames: (1-alpha)*a + alpha*b."""
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    blended = (1.0 - alpha) * aa + alpha * bb
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def stitch(process_fn, frame_iter, window, overlap, on_emit):
    """
    Processes `frame_iter` in overlapping windows and streams stitched output
    frames to `on_emit(frame)`, one at a time.

    Parameters:
      process_fn (callable) - list[frame] -> list[frame] (same length)
      frame_iter (iterable) - source frames (a streaming generator is fine)
      window (int)          - frames per window
      overlap (int)         - frames shared/blended between adjacent windows
      on_emit (callable)    - receives each finished output frame in order

    Returns: total number of frames emitted
    """
    stride = window - overlap
    if overlap < 0 or overlap * 2 > window:
        raise ValueError(
            f"overlap ({overlap}) must be between 0 and half of window_size ({window})"
        )

    prev_tail = None  # last `overlap` output frames of the previous window
    emitted = 0

    def emit(frame):
        nonlocal emitted
        on_emit(frame)
        emitted += 1

    for win_frames, pad in iter_windows(frame_iter, window, stride):
        out = process_fn(win_frames)
        if pad:
            out = out[: len(out) - pad]
        length = len(out)

        if prev_tail is None:
            # first window: emit everything except the tail that may blend next
            split = max(0, length - overlap)
            for frame in out[:split]:
                emit(frame)
            prev_tail = out[split:]
        else:
            # blend this window's head against the previous window's held tail
            held = len(prev_tail)
            for j in range(held):
                emit(_blend(prev_tail[j], out[j], (j + 1) / (held + 1)))
            # emit the middle, then hold the tail for the next window. Clamp the
            # boundary so a short final window (length < 2*overlap) can't re-emit
            # frames already covered by the head blend above.
            mid_end = max(held, length - overlap)
            for frame in out[held:mid_end]:
                emit(frame)
            prev_tail = out[mid_end:]

    if prev_tail:
        for frame in prev_tail:
            emit(frame)

    return emitted
