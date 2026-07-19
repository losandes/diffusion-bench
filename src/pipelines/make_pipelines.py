from constants import GENERATOR, REFINER, UPSCALER, VID2VID

from . import analog_diffusion as analog
from . import animatediff_vid2vid as animatediff
from . import analog_diffusion_euler_a as analog_euler
from . import deci_diffusion as dd
from . import dreamlike_photoreal as dp
from . import hidream_e1 as hidreamE1
from . import hidream_i1 as hidreamI1
from . import openjourney as oj
from . import pix2pix as p2p
from . import stable_diffusion_2_1 as sdxl_2_1
from . import stable_diffusion_2_depth as sdxl_2_depth
from . import stable_diffusion_x4_upscaler as sdx4_upscaler
from . import stable_diffusion_xl_base as sdxl_base
from . import stable_diffusion_xl_lineart as sdxl_lineart
from . import stable_diffusion_xl_refiner as sdxl_refiner
from . import wrong_lora_refiner as wrong
from . import wuerstchen

PIPELINES = {
    "guoyww/animatediff-v1-5-2": {
        "short_name": "animatediff",
        "type": VID2VID,  # coherent video-to-video (needs a video input)
        "factory": animatediff.makePipelines,
        # AnimateDiff's UNetMotionModel has no plain `.unet.config.in_channels`
        # read path we rely on; override to skip it. Latents are managed by the
        # pipeline (video frames), not the img2img latents helper.
        "in_channels": 4,
        "supports_latents": False,
    },
    "HiDream-ai/HiDream-I1-Full": {
        "short_name": "hidream-i1",
        "type": GENERATOR,
        "factory": hidreamI1.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "HiDream-ai/HiDream-E1-Full": {
        "short_name": "hidream-e1",
        "type": REFINER,  # editing / any-to-any model: needs an input image (img2img)
        "factory": hidreamE1.makePipelines,
        "in_channels": None,
        "supports_latents": False,
    },
    "wavymulder/Analog-Diffusion": {
        "short_name": "analog",
        "type": GENERATOR,
        "factory": analog.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "wavymulder/Analog-Diffusion/EulerA": {
        "short_name": "analog_euler",
        "type": GENERATOR,
        "factory": analog_euler.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "Deci/DeciDiffusion-v1-0": {
        "short_name": "ddv1",
        "type": GENERATOR,
        "factory": dd.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "dreamlike-art/dreamlike-photoreal-2.0": {
        "short_name": "dp2",
        "type": GENERATOR,
        "factory": dp.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "prompthero/openjourney": {
        "short_name": "oj",
        "type": GENERATOR,
        "factory": oj.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "stabilityai/stable-diffusion-2-1": {
        "short_name": "sdxl_2_1",
        "type": GENERATOR,
        "factory": sdxl_2_1.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "stabilityai/stable-diffusion-2-depth": {
        "short_name": "sdxl_2_depth",
        "type": REFINER,
        "factory": sdxl_2_depth.makePipelines,
        "in_channels": None,
        "supports_latents": False,
    },
    "stabilityai/stable-diffusion-x4-upscaler": {
        "short_name": "sdx4-up",
        "type": UPSCALER,
        "factory": sdx4_upscaler.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "stabilityai/stable-diffusion-xl-base-1.0": {
        "short_name": "sdxl_base1",
        "type": GENERATOR,
        "factory": sdxl_base.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "stabilityai/stable-diffusion-xl-refiner-1.0": {
        "short_name": "sdxl_refiner1",
        "type": REFINER,
        "factory": sdxl_refiner.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "TencentARC/t2i-adapter-lineart-sdxl-1.0": {
        "short_name": "sdxl_lineart1",
        "type": REFINER,
        "factory": sdxl_lineart.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "timbrooks/instruct-pix2pix": {
        "short_name": "p2p",
        "type": REFINER,
        "factory": p2p.makePipelines,
        "in_channels": 4,  # override the in_channels
        "supports_latents": True,
    },
    "minimaxir/sdxl-wrong-lora": {
        "short_name": "wront-lora",
        "type": GENERATOR,
        "factory": wrong.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
    "warp-diffusion/wuerstchen": {
        "short_name": "wuerstchen",
        "type": GENERATOR,
        "factory": wuerstchen.makePipelines,
        "in_channels": None,
        "supports_latents": True,
    },
}
singletons = {}


def make_one_pipeline(name, device):
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
        pipe = PIPELINES[name]["factory"](device)[1]
        in_channels = None

        if PIPELINES[name]["in_channels"] is not None:
            in_channels = PIPELINES[name]["in_channels"]
        else:
            in_channels = pipe.unet.config.in_channels

        singletons[name] = {
            "name": name,
            "short_name": PIPELINES[name]["short_name"],
            "type": PIPELINES[name]["type"],
            "factory": PIPELINES[name]["factory"],
            "pipe": pipe,
            "in_channels": in_channels,
            "supports_latents": PIPELINES[name]["supports_latents"],
        }

        return singletons[name]
    else:
        raise Exception(f"A pipeline with name, {name}, is not registered")


def make_pipelines(names, device):
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
