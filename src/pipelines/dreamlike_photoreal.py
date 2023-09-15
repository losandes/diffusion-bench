from diffusers import StableDiffusionPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0

  This model was trained on 768x768px images, so use:
  - 768x768px
  - 640x896px
  - 896x640px
  - 768x1024px
  - 1024x768px
  """
  model_id = "dreamlike-art/dreamlike-photoreal-2.0"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
  ).to(device)

  return [model_id, pipe]
