from diffusers import StableDiffusionPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/prompthero/openjourney
  """
  model_id = "prompthero/openjourney"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
  ).to(device)

  return [model_id, pipe]
