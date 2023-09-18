from files.load import load_image

def _refine (
  options,
  **kwargs,
):
  """
  Refines, upscales, or otherwise transforms an image
  """
  model = options['model']
  prompt = options['prompt']
  in_paths = options['input_paths']
  out_paths = options['output_paths']
  latents = options['latents']
  seeds = options['seeds']
  images = []

  for i in range(len(out_paths)):
    print("")
    print(f"model_id: {model['name']}")
    print(f"prompt: {prompt}")
    print(f"kwargs: {kwargs}")
    print(f"seed:   {seeds[i]}")
    print(f"image:  {out_paths[i]}")
    print("")

    image = load_image(in_paths[i])
    refined = model['pipe'](
      prompt,
      image=image,
      latents=latents[i],
      **kwargs,
    ).images[0]
    images.append([out_paths[i], refined])
    refined.save(out_paths[i])

  return images

def img2img (options):
  """
  Refines, upscales, or otherwise transforms an image

  Parameters:
    options (dict): the arguments passed to main, with defaults added

  Returns: [Image]
  """
  return _refine(
    options,
    negative_prompt=options['negative_prompt'],
    num_inference_steps=options['steps'],
  )
