from app.get_arr_value import get_value_or_none, get_value_or_first
from files.load import load_image
from pipelines.make_latents import make_latents

def _refine (
  options,
  step,
  **kwargs,
):
  """
  Refines, upscales, or otherwise transforms an image
  """
  pipeline = options['MODELS'][step][1]
  prompt = get_value_or_first(options['PROMPT'], step)
  in_paths = None
  out_paths = options['PATHS'][step]
  device = options['DEVICE']
  seeds = seeds=options['SEEDS']
  height = options['HEIGHT']
  width = options['WIDTH']
  use_custom_latents = options['CUSTOM_LATENTS']
  [model_id, pipe] = pipeline
  images = []

  if step == 0:
    in_paths = options['INPUT_PATHS']
  else:
    in_paths = options['PATHS'][step - 1]

  for i in range(len(out_paths)):
    latent_seed = None
    latents = None

    if use_custom_latents == True:
      [latent_seed, latents] = make_latents(
        device,
        # TODO this is a temporary hack to get latents working with pix2pix
        pipe.unet.config.in_channels if model_id != "timbrooks/instruct-pix2pix" else 4,
        height=height,
        width=width,
        seed=get_value_or_none(seeds, i),
      )

    print("")
    print(f"model_id: {model_id}")
    print(f"prompt: {prompt}")
    print(f"kwargs: {kwargs}")
    print(f"seed:   {latent_seed}")
    print("")

    image = load_image(in_paths[i])
    refined = pipe(
      prompt,
      image=image,
      latents=latents,
      **kwargs,
    ).images[0]
    images.append([out_paths[i], refined])
    refined.save(out_paths[i])

  return images

def img2img (options, step):
  """
  Refines, upscales, or otherwise transforms an image

  Parameters:
    options (dict): the arguments passed to main, with defaults added
    step (int): if this is one of many images being generated, the
                zero-based index of this iteration

  Returns: [Image]
  """
  return _refine(
    options,
    step,
    negative_prompt=get_value_or_first(options['NEGATIVE_PROMPT'], step),
    num_inference_steps=options['STEPS'][step],
  )
