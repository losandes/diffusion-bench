def _generate (
  options,
  **kwargs,
):
  model = options['model']
  prompt = options['prompt']
  latents = options['latents']
  seeds = options['seeds']
  out_paths = options['output_paths']
  images = []

  for i in range(len(out_paths)):
    print("")
    print(f"model_id: {model['name']}")
    print(f"prompt: {prompt}")
    print(f"kwargs: {kwargs}")
    print(f"seed:   {seeds[i]}")
    print(f"image:  {out_paths[i]}")
    print("")

    image = model['pipe'](
      prompt,
      latents=latents[i],
      **kwargs,
    ).images[0]
    images.append([out_paths[i], image])
    image.save(out_paths[i])

  return images

def txt2img (options):
  """
  Generates an image

  Parameters:
    options (dict): the arguments passed to main, with defaults added

  Returns: [Image]
  """
  return _generate(
    options,
    negative_prompt=options['negative_prompt'],
    width=options['width'],
    height=options['height'],
    num_inference_steps=options['steps'],
  )
