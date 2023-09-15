from diffusers import StableDiffusionUpscalePipeline

def makePipelines (device):
  """
  @see https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
  """
  model_id = "stabilityai/stable-diffusion-x4-upscaler"
  pipe = StableDiffusionUpscalePipeline.from_pretrained(
    model_id,
  ).to(device)

  return [model_id, pipe]
