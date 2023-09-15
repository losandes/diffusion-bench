from diffusers import DiffusionPipeline

def makePipelines (device):
  """
  @see https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
  @see https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
  """
  base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
  base = DiffusionPipeline.from_pretrained(
    base_model_id,
    use_safetensors=True,
  ).to(device)

  refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
  refiner = DiffusionPipeline.from_pretrained(
    refiner_model_id,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    use_safetensors=True,
  ).to(device)

  return [[base_model_id, base], [refiner_model_id, refiner]]
