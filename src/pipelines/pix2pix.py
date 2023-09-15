from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def makePipelines (device):
  """
  @see https://huggingface.co/timbrooks/instruct-pix2pix
  """
  model_id = "timbrooks/instruct-pix2pix"
  pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    model_id,
    safety_checker=None
  ).to(device)
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

  return [model_id, pipe]
