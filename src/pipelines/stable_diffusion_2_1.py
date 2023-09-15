from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def makePipelines (device):
  """
  @see https://huggingface.co/stabilityai/stable-diffusion-2-1
  """
  model_id = "stabilityai/stable-diffusion-2-1"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    use_safetensors=True,
  )
  pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
  pipe = pipe.to(device)

  return [model_id, pipe]
