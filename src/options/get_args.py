import argparse
from constants import GENERATOR
from options.get_arr_value import get_value_or_first, get_value_or_none
from options.get_device_type import get_device_type
from options.make_paths import make_paths
from options.split_to_list import split_to_list
from pipelines.make_pipelines import make_one_pipeline
from pipelines.make_latents import make_latents

def parse_args ():
  """
  Parses the args passed in the terminal command
  """
  # Initialize parser
  parser = argparse.ArgumentParser()

  # Adding optional argument
  parser.add_argument("-p", "--prompt", help = "A pipe-delimited list of descriptions of what you would like to render")
  parser.add_argument("-n", "--negative_prompt", help = "A pipe-delimited list of descriptions of what you would NOT like to render (default=None)")
  parser.add_argument("-x", "--width", help = "The width of the output image (default=896)")
  parser.add_argument("-y", "--height", help = "The height of the output image (default=640)")
  parser.add_argument("-c", "--count", help = "The number of images to produce with each given model (default=1)")
  parser.add_argument("-s", "--seeds", help = "A comma-separated list of PRNGs to use when generating an image to produce more predictable results and explore an idea")
  parser.add_argument("-m", "--models", help = "A comma-separated list of HuggingFace Models to to use. By default, dreamlike-art/dreamlike-photoreal-2.0 is used to generate an image followed by 3, sequential steps of refinement using stabilityai/stable-diffusion-xl-refiner-1.0")
  parser.add_argument("-l", "--custom_latents", help = "A comma-separated list of booleans: whether or not to use custom latents (seeds) for each model")
  parser.add_argument("-r", "--steps", help = "A comma-separated list of the number inference steps to use with each model (length must match the length of -m)")
  parser.add_argument("-o", "--output_path", help = "A path to a folder where images will be saved (can be relative)")
  parser.add_argument("-t", "--output_path_template", help = "A template for naming the files (default=\":path/:count_idx-:type-:model_idx.png\")")
  parser.add_argument("-i", "--input_paths", help = "A comma-separated list of paths to images that will be refined or upscaled by the given models")
  parser.add_argument("-a", "--refinement_mode", help = "one of: \"sequence\" (each pass is fed into the next pass), \"first_to_many\" (each pass is fed the first item generated), or \"in_to_many\" (each pass is fed the value of -i/--input_paths)")
  parser.add_argument("-d", "--device_type", help = "The type of device the pipes will be fed to for processing (default=\"cuda\" if cuda is supported, else \"mps\" if apple M1/M2, else \"cpu\")")

  # TODO: Add refinement_mode selection
  # - "sequence" (each pass is fed into the next pass)
  # - "first_to_many" (each pass is fed the first item generated)
  # - "in_to_many" (each pass is fed the value of -i/--input_paths)

  # TODO: Add option to turn off custom latents

  # Read arguments from command line
  return parser.parse_args()

def with_args (**kwargs):
  """
    Validates arguments, prepares the model, and returns an arg model

    Parameters:
      model_id (string)
      output_path (string)
      prompt (string)
      negative_prompt (string?)
      width (int?)
      height (int?)
      count (int?)
      seed (string?)
      custom_latents (boolean?)
      steps (int?)
      input_path (string?)
      device (string?)
  """
  if kwargs['prompt'] is None:
    print(f"expected string, actual prompt: {kwargs['prompt']}")
    raise Exception("At least one prompt is required")

  if kwargs['model_id'] is None:
    print(f"expected string[], actual model_id: {kwargs['model_id']}")
    raise Exception("model_id(s) are required")

  if kwargs['output_paths'] is None:
    print(f"expected string[], actual output_paths: {kwargs['output_paths']}")
    print(f"actual model_id: {kwargs['model_id']}")
    raise Exception("output_paths are required and must be an array the same length as the model_ids")

  width = int(kwargs['width'] if kwargs['width'] is not None else 896)
  height = int(kwargs['height'] if kwargs['height'] is not None else 640)
  device_type = kwargs['device_type'] if kwargs['device_type'] is not None else get_device_type()
  model = make_one_pipeline(kwargs['model_id'], device_type)
  custom_latents = str(kwargs['custom_latents']).lower() != "false"
  seed = int(kwargs['seed']) if kwargs['seed'] is not None else None

  input_paths = kwargs['input_paths']

  if input_paths == None and model['type'] != GENERATOR and kwargs['previous_pass'] is not None:
    input_paths = kwargs['previous_pass']['output_paths']

  seeds = []
  latents = []
  output_paths = []

  for _idx, path in enumerate(kwargs['output_paths']):
    _latent_seed = None
    _latents = None
    path = path.replace(":name", model['short_name'])

    if custom_latents == True:
      [_latent_seed, _latents] = make_latents(
        device_type,
        model['in_channels'],
        height=height,
        width=width,
        seed=seed,
      )
      seeds.append(_latent_seed)
      latents.append(_latents)
      output_paths.append(path.replace(":seed", str(_latent_seed)))
    else:
      seeds.append(None)
      latents.append(None)
      output_paths.append(path.replace(":seed", "image"))

  return {
    "prompt": kwargs['prompt'],
    "negative_prompt": kwargs['negative_prompt'],
    "width": width,
    "height": height,
    "model_id": kwargs['model_id'],
    "model": model,
    "latents": latents,
    "seeds": seeds,
    "steps": int(kwargs['steps'] if kwargs['steps'] is not None else 10),
    "input_paths": input_paths,
    "output_paths": output_paths,
    "device_type": device_type,
  }

def map_terminal_input (args):
  """
  Maps terminal input to args for img2img and txt2img
  """

  DEVICE = args.device_type if args.device_type is not None else get_device_type()
  PROMPT = split_to_list("|")(args.prompt)
  NEGATIVE_PROMPT = split_to_list("|")(args.negative_prompt)
  WIDTH = split_to_list(",")(args.width if args.width is not None else "896")
  HEIGHT = split_to_list(",")(args.height if args.height is not None else "640")
  COUNT = int(args.count if args.count is not None else 1)
  SEEDS = split_to_list(",")(args.seeds)
  MODEL_IDS = split_to_list(",")(
    args.models if args.models is not None
    else "dreamlike-art/dreamlike-photoreal-2.0"
  )
  STEPS = split_to_list(",")(
    args.steps if args.steps is not None else "10"
  )
  INPUT_PATHS = split_to_list(",")(args.input_paths)
  OUTPUT_PATH = args.output_path if args.output_path is not None else "images"
  OUTPUT_PATH_TEMPLATE = args.output_path_template if args.output_path_template is not None else ":path/:count_idx-:model_idx-:name-:seed.png"
  OUTPUT_PATHS = []
  CUSTOM_LATENTS = split_to_list(",")(args.custom_latents if args.custom_latents is not None else "True")

  for i in range(len(MODEL_IDS)):
    OUTPUT_PATHS.append(make_paths(OUTPUT_PATH_TEMPLATE)(OUTPUT_PATH, i, COUNT))

  # if MODELS[0]['type'] is not GENERATOR and len(INPUT_PATHS) == 0:
  #   raise Exception("Input paths (-i or --input_paths) are required when the first model is not a generator")

  if len(MODEL_IDS) != len(STEPS):
    raise Exception("The models (-m or --models) and steps (-s or --steps) lists must be of the same length")

  passes = []

  for idx, _model_id in enumerate(MODEL_IDS):
    previous_pass = None if idx == 0 else passes[idx - 1]

    item = with_args(
      prompt=get_value_or_first(PROMPT, idx),
      negative_prompt=get_value_or_first(NEGATIVE_PROMPT, idx),
      width=get_value_or_first(WIDTH, idx),
      height=get_value_or_first(HEIGHT, idx),
      count=COUNT,
      seed=get_value_or_none(SEEDS, idx),
      model_id=get_value_or_none(MODEL_IDS, idx),
      custom_latents=get_value_or_first(CUSTOM_LATENTS, idx),
      steps=get_value_or_none(STEPS, idx),
      input_paths=get_value_or_none(INPUT_PATHS, idx),
      output_paths=get_value_or_none(OUTPUT_PATHS, idx),
      device_type=DEVICE,
      previous_pass=previous_pass,
    )
    passes.append(item)
    print("")
    print(f"{str(idx + 1).zfill(2)} =================================================")
    print(f"prompt:          {item['prompt']}")
    print(f"negative_prompt: {item['negative_prompt']}")
    print(f"width:           {item['width']}")
    print(f"height:          {item['height']}")
    print(f"seeds:           {item['seeds']}")
    print(f"model_id:        {item['model_id']}")
    print(f"steps:           {item['steps']}")
    print(f"input_paths:     {item['input_paths']}")
    print(f"output_paths:    {item['output_paths']}")
    print(f"device_type:     {item['device_type']}")
    print("====================================================")
    print("")

  return passes
