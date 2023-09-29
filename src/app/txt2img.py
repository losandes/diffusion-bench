from ..files.save import save_image

def _generate (
  options,
  ensembleIdx,
  **kwargs,
):
  copyright = options['copyright']
  models = options['models']
  model = options['model']
  prompt = options['prompt']
  latents = options['latents']
  seeds = options['seeds']
  out_paths = options['output_paths']
  images = []

  for i in range(len(out_paths)):
    print("")
    print(f"kwargs:        {kwargs}")
    print(f"prompt:        {prompt}")
    print(f"model_id:      {model['name']}")
    print(f"seed:          {seeds[i]}")
    print(f"output_image:  {out_paths[i]}")
    print("")

    image = model['pipe'](
      prompt,
      latents=latents[i],
      **kwargs,
    ).images[0]
    images.append([out_paths[i], image])

    if out_paths[i] is not None:
      save_image(image, out_paths[i], prompt, seeds[i], copyright, models, ensembleIdx)

  return images

def txt2img (options, ensembleIdx = 0):
  """
  Generates an image

  Parameters:
    options (dict): the arguments passed to main, with defaults added
    ensembleIdx (int): the index of the step, when this is one step in an ensemble

  Returns: [Image]
  """
  return _generate(
    options,
    ensembleIdx,
    negative_prompt=options['negative_prompt'],
    width=options['width'],
    height=options['height'],
    num_inference_steps=options['steps'],
  )
