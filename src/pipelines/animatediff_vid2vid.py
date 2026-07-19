import torch
from diffusers import (
    AnimateDiffVideoToVideoPipeline,
    AutoencoderKL,
    DDIMScheduler,
    MotionAdapter,
)


def makePipelines(device):
    """
    AnimateDiff video-to-video: transforms an existing clip with a text prompt
    while preserving motion via a MotionAdapter (temporal layers), so output is
    coherent frame-to-frame rather than flickering like naive per-frame img2img.

    @see https://huggingface.co/docs/diffusers/api/pipelines/animatediff

    Built on SD1.5, so it expects ~512px input (downscale the 896x640 default
    via -x/-y). The app enables FreeNoise per-run for long-range coherence.

    NOTE (Apple Silicon): runs in float16 with attention/vae slicing to fit in
    unified memory. If frames come out black on MPS, try float32 instead.
    """
    model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
    adapter_id = "guoyww/animatediff-motion-adapter-v1-5-2"
    vae_id = "stabilityai/sd-vae-ft-mse"  # Realistic Vision ships "noVAE"

    dtype = torch.float16

    adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)

    pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        vae=vae,
        torch_dtype=dtype,
    ).to(device)

    # Scheduler settings recommended for AnimateDiff (linear beta, linspace).
    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )

    # For M1/M2, reduce memory pressure so the sequence of frames fits.
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    return [model_id, pipe]
