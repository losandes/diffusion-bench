from diffusers import StableDiffusionPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/Deci/DeciDiffusion-v1-0
  @see https://deci.ai/blog/decidiffusion-1-0-3x-faster-than-stable-diffusion-same-quality/
  """
  model_id = "Deci/DeciDiffusion-v1-0"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    custom_pipeline=model_id,
    use_safetensors=True,
  )
  pipe.unet = pipe.unet.from_pretrained(
    model_id,
    subfolder='flexible_unet',
  )

  pipe = pipe.to(device)

  return [model_id, pipe]
