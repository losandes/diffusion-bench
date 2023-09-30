from diffusers import DiffusionPipeline, AutoencoderKL

def makePipelines (device):
  """
  @see https://huggingface.co/minimaxir/sdxl-wrong-lora

  NOTE: you have to use "wrong" as a negative prompt for this to take effect
  """
  pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    use_safetensors=True
  )

  pipe.load_lora_weights("minimaxir/sdxl-wrong-lora")
  pipe = pipe.to(device)

  return ["stabilityai/stable-diffusion-xl-base-1.0, minimaxir/sdxl-wrong-lora", pipe]
