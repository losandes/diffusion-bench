from . import openjourney as oj
from . import dreamlike_photoreal as dp
from . import stable_diffusion_2_1 as sdxl_2_1
from . import stable_diffusion_xl_base as sdxl_base
from . import stable_diffusion_xl_refiner as sdxl_refiner
from . import stable_diffusion_xl_lineart as sdxl_lineart
from . import stable_diffusion_x4_upscaler as sdx4_upscaler
from . import pix2pix as p2p

from constants import GENERATOR, REFINER, UPSCALER

PIPELINES = {
  "dreamlike-art/dreamlike-photoreal-2.0": {
    "short_name": "dp2",
    "type": GENERATOR,
    "factory": dp.makePipelines,
    "in_channels": None,
  },
  "stabilityai/stable-diffusion-xl-base-1.0": {
    "short_name": "sdxl_base1",
    "type": GENERATOR,
    "factory": sdxl_base.makePipelines,
    "in_channels": None,
  },
  "stabilityai/stable-diffusion-2-1": {
    "short_name": "sdxl_2_1",
    "type": GENERATOR,
    "factory": sdxl_2_1.makePipelines,
    "in_channels": None,
  },
  "prompthero/openjourney": {
    "short_name": "oj",
    "type": GENERATOR,
    "factory": oj.makePipelines,
    "in_channels": None,
  },

  "stabilityai/stable-diffusion-xl-refiner-1.0": {
    "short_name": "sdxl_refiner1",
    "type": REFINER,
    "factory": sdxl_refiner.makePipelines,
    "in_channels": None,
  },
  "TencentARC/t2i-adapter-lineart-sdxl-1.0": {
    "short_name": "sdxl_lineart1",
    "type": REFINER,
    "factory": sdxl_lineart.makePipelines,
    "in_channels": None,
  },
  "timbrooks/instruct-pix2pix": {
    "short_name": "p2p",
    "type": REFINER,
    "factory": p2p.makePipelines,
    "in_channels": 4, # override the in_channels
  },

  "stabilityai/stable-diffusion-x4-upscaler": {
    "short_name": "sdx4-up",
    "type": UPSCALER,
    "factory": sdx4_upscaler.makePipelines,
    "in_channels": None,
  },
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
    pipe = PIPELINES[name]['factory'](device)[1]
    in_channels = None

    if PIPELINES[name]['in_channels'] is not None:
      in_channels = PIPELINES[name]['in_channels']
    else:
      in_channels = pipe.unet.config.in_channels

    singletons[name] = {
      "name": name,
      "short_name": PIPELINES[name]['short_name'],
      "type": PIPELINES[name]['type'],
      "factory": PIPELINES[name]['factory'],
      "pipe": pipe,
      "in_channels": in_channels,
    }

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
