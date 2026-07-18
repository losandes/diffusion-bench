from diffusers import StableDiffusionDepth2ImgPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/stabilityai/stable-diffusion-2-depth
  """
  model_id = "stabilityai/stable-diffusion-2-depth"
  pipe = StableDiffusionDepth2ImgPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
  ).to(device)

  return [model_id, pipe]
