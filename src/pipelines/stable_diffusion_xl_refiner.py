from diffusers import StableDiffusionXLImg2ImgPipeline

def makePipelines (device): # , base_pipeline):
  """
  @see https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
  """
  model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
  pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
    # text_encoder_2=base_pipeline.text_encoder_2,
    # vae=base_pipeline.vae,
  ).to(device)

  return [model_id, pipe]
