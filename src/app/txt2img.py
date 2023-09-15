from app.get_arr_value import get_value_or_none, get_value_or_first
from pipelines.make_latents import make_latents

def _generate (
  options,
  step,
  **kwargs,
):
  pipeline = options['MODELS'][step][1]
  prompt = get_value_or_first(options['PROMPT'], step)
  out_paths = options['PATHS'][step]
  device = options['DEVICE']
  seeds = seeds=options['SEEDS']
  [model_id, pipe] = pipeline
  images = []

  for i in range(len(out_paths)):
    [latent_seed, latents] = make_latents(
      device,
      pipe.unet.config.in_channels,
      height=kwargs['height'],
      width=kwargs['width'],
      seed=get_value_or_none(seeds, i),
    )

    print("")
    print(f"model_id: {model_id}")
    print(f"prompt: {prompt}")
    print(f"kwargs: {kwargs}")
    print(f"seed:   {latent_seed}")
    print("")

    image = pipe(
      prompt,
      latents=latents,
      **kwargs,
    ).images[0]
    images.append([out_paths[i], image])
    image.save(out_paths[i])

  return images

def txt2img (options, step=0):
  """
  Generates an image

  Parameters:
    options (dict): the arguments passed to main, with defaults added
    step (int): if this is one of many images being generated, the
                zero-based index of this iteration

  Returns: [Image]
  """
  return _generate(
    options,
    step,
    negative_prompt=get_value_or_first(options['NEGATIVE_PROMPT'], step),
    num_inference_steps=options['STEPS'][step],
    width=options['WIDTH'],
    height=options['HEIGHT'],
  )
