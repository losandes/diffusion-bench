from diffusers import StableDiffusionXLAdapterPipeline, T2IAdapter, EulerAncestralDiscreteScheduler, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from controlnet_aux.lineart import LineartDetector

def makePipelines (device):
  """
  @see https://huggingface.co/TencentARC/t2i-adapter-lineart-sdxl-1.0
  """
  adapter_id = "TencentARC/t2i-adapter-lineart-sdxl-1.0"
  # load adapter
  adapter = T2IAdapter.from_pretrained(
    adapter_id,
  ).to(device)

  # load euler_a scheduler
  model_id = "stabilityai/stable-diffusion-xl-base-1.0"
  euler_a = EulerAncestralDiscreteScheduler.from_pretrained(
    model_id,
    subfolder="scheduler"
  )
  vae=AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
  )
  pipe = StableDiffusionXLAdapterPipeline.from_pretrained(
      model_id,
      vae=vae,
      adapter=adapter,
      scheduler=euler_a,
  ).to(device)
  pipe.enable_xformers_memory_efficient_attention()

  line_detector = LineartDetector.from_pretrained(
    "lllyasviel/Annotators"
  ).to(device)

  return [adapter_id, line_detector, pipe]
