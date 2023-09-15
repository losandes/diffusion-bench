import pipelines.openjourney as oj
import pipelines.dreamlike_photoreal as dp
import pipelines.stable_diffusion_2_1 as sdxl_2_1
import pipelines.stable_diffusion_xl_base as sdxl_base
import pipelines.stable_diffusion_xl_refiner as sdxl_refiner
import pipelines.stable_diffusion_xl_lineart as sdxl_lineart
import pipelines.stable_diffusion_x4_upscaler as sdx4_upscaler
import pipelines.pix2pix as p2p

from constants import GENERATOR, REFINER, UPSCALER

PIPELINES = {
  "dreamlike-art/dreamlike-photoreal-2.0": [GENERATOR, dp.makePipelines],
  "stabilityai/stable-diffusion-xl-base-1.0": [GENERATOR, sdxl_base.makePipelines],
  "stabilityai/stable-diffusion-2-1": [GENERATOR, sdxl_2_1.makePipelines],
  "prompthero/openjourney": [GENERATOR, oj.makePipelines],

  "stabilityai/stable-diffusion-xl-refiner-1.0": [REFINER, sdxl_refiner.makePipelines],
  "TencentARC/t2i-adapter-lineart-sdxl-1.0": [REFINER, sdxl_lineart.makePipelines],
  "timbrooks/instruct-pix2pix": [REFINER, p2p.makePipelines],

  "stabilityai/stable-diffusion-x4-upscaler": [UPSCALER, sdx4_upscaler.makePipelines],
}
singletons = {}

def make_one_pipeline (name, device):
  """
  Creates an instance of a pipeline

  Usage:
    pipe = make_one_pipeline("dreamlike-art/dreamlike-photoreal-2.0", "mps")

  Returns: [string, [model_id, pipe]]
  """
  if name in singletons:
    # print(f"reusing: {name}")
    return singletons[name]
  elif name in PIPELINES:
    print(f"loading pipeline: {name}")
    singletons[name] = [PIPELINES[name][0], PIPELINES[name][1](device)]
    return singletons[name]
  else:
    raise Exception(f"A pipeline with name, {name}, is not registered")

def make_pipelines (names, device):
  """
  Foreach name, creates or gets an instance of a pipeline

  Usage:
    pipe = make_pipelines([
      "dreamlike-art/dreamlike-photoreal-2.0",
      "stabilityai/stable-diffusion-xl-refiner-1.0",
    ], "mps")

  Returns: [[string, [model_id, pipe]]]
  """
  pipelines = []

  for idx, name in enumerate(names):
    pipelines.append(make_one_pipeline(name, device))

  return pipelines
