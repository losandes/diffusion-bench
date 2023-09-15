from diffusers import DiffusionPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
  """
  model_id = "stabilityai/stable-diffusion-xl-base-1.0"
  pipe = DiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
  ).to(device)

  return [model_id, pipe]
