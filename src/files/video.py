"""
Streaming video I/O for prompt-driven video-to-video.

No diffusion pipeline consumes an mp4 container directly: we decode the file to
RGB frames (PIL), let a model transform them, then re-encode. To support
arbitrary-length video without exhausting memory, reading is a generator and
writing is incremental -- only the frames in the current processing window are
ever held in memory.

Backed by imageio + imageio-ffmpeg (ffmpeg is the decode/encode engine).
"""
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def probe(path):
    """
    Reads container metadata without decoding the whole video.

    Returns a dict: { "fps": float, "size": (w, h), "nframes": int|None }
    nframes is None when the container doesn't report a reliable count.
    """
    reader = imageio.get_reader(path)
    try:
        meta = reader.get_meta_data()
    finally:
        reader.close()

    nframes = meta.get("nframes")
    # imageio reports inf (or a sentinel) when the count is unknown.
    if nframes is None or nframes != nframes or nframes == float("inf"):
        nframes = None
    else:
        nframes = int(nframes)

    return {
        "fps": float(meta.get("fps", 0) or 0),
        "size": tuple(meta.get("size")) if meta.get("size") else None,
        "nframes": nframes,
    }


def read_frames(path, fps=None, max_frames=None, size=None):
    """
    Streams RGB PIL frames from a video file.

    Parameters:
      path (str)          - the video file to decode
      fps (float?)        - target frame rate; when lower than the source, frames
                            are subsampled to approximate it (never upsamples)
      max_frames (int?)   - stop after yielding this many frames
      size ((w, h)?)      - resize each frame (diffusion models are resolution
                            sensitive; AnimateDiff/SD1.5 want ~512px)

    Yields: PIL.Image (RGB)
    """
    meta = probe(path)
    source_fps = meta["fps"]
    reader = imageio.get_reader(path)

    emitted = 0
    try:
        for i, frame in enumerate(reader):
            # Subsample toward the target fps: emit frame i only when it advances
            # the output timeline. Guards against source_fps being 0/unknown.
            if fps is not None and source_fps and fps < source_fps:
                if int(i * fps / source_fps) < emitted:
                    continue

            image = Image.fromarray(frame).convert("RGB")
            if size is not None:
                image = image.resize(size, Image.LANCZOS)

            yield image
            emitted += 1

            if max_frames is not None and emitted >= max_frames:
                break
    finally:
        reader.close()


class VideoWriter:
    """
    Incremental video writer -- append one frame at a time so output frames can
    be flushed to disk as each processing window completes.

    Usage:
      with VideoWriter("out.mp4", fps=16) as writer:
          for frame in frames:
              writer.append(frame)
    """

    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        # macro_block_size=None avoids forcing dimensions to a multiple of 16
        # (otherwise imageio silently resizes odd resolutions).
        self._writer = imageio.get_writer(path, fps=fps, macro_block_size=None)

    def append(self, frame):
        """Append a single PIL RGB frame."""
        if isinstance(frame, Image.Image):
            frame = np.asarray(frame.convert("RGB"))
        self._writer.append_data(frame)

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_sidecar(video_path, meta):
    """
    Writes a JSON sidecar next to the output video.

    mp4 has no reliable, portable place to stamp generation parameters the way
    EXIF does for images (see files/save.py), so provenance -- prompt, model,
    seed, steps, strength, fps, source -- lives in <video_path>.json instead.
    """
    sidecar_path = f"{video_path}.json"
    with open(sidecar_path, "w") as handle:
        json.dump(meta, handle, indent=2, default=str)
    return sidecar_path
