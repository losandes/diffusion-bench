from diffusers import EulerAncestralDiscreteScheduler
from . import analog_diffusion

def makePipelines (device):
  """
  @see https://huggingface.co/wavymulder/Analog-Diffusion
  @see https://huggingface.co/wavymulder/Analog-Diffusion/resolve/main/parameters_used_examples.txt

  NOTE: You have to use "analog style" in the prompt for this to take effect
  """
  [model_id, pipe] = analog_diffusion.makePipelines(device)
  pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

  return [model_id, pipe]
