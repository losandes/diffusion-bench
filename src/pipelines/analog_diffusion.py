from diffusers import StableDiffusionPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/wavymulder/Analog-Diffusion
  @see https://huggingface.co/wavymulder/Analog-Diffusion/resolve/main/parameters_used_examples.txt

  NOTE: You have to use "analog style" in the prompt for this to take effect
  """
  model_id = "wavymulder/Analog-Diffusion"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
  ).to(device)

  return [model_id, pipe]
