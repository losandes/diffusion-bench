import argparse
from ..constants import GENERATOR
from ..pipelines.make_pipelines import make_one_pipeline
from ..pipelines.make_latents import make_latents
from .get_arr_value import get_value_or_first, get_value_or_none
from .get_device_type import get_device_type
from .make_paths import make_all_paths, make_empty_paths, maybe_update_path, add_name_to_path, add_seed_to_path
from .split_to_list import split_to_list

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
  parser.add_argument("-d", "--device_type", help = "The type of device the pipes will be fed to for processing (default=\"cuda\" if cuda is supported, else \"mps\" if apple M1/M2, else \"cpu\")")
  parser.add_argument("--refinement_mode", help = "one of: \"sequence\" (each pass is fed into the next pass), \"first_to_many\" (each pass is fed the first item generated), or \"in_to_many\" (each pass is fed the value of -i/--input_paths)")
  parser.add_argument("--copyright", help = "Set who should be listed as the copyright owner of the images that are created")

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
      prompt (string)
      negative_prompt (string?)
      width (int?)
      height (int?)
      count (int?)
      seed (string?)
      custom_latents (boolean?)
      steps (int?)
      input_paths (string[]?)
      output_paths (string[]?)
      device (string?)
  """
  if not 'prompt' in kwargs or kwargs['prompt'] is None:
    print(f"expected string, actual prompt: {kwargs['prompt']}")
    raise Exception("At least one prompt is required")

  if not 'model_id' in kwargs or kwargs['model_id'] is None:
    print(f"expected string[], actual model_id: {kwargs['model_id']}")
    raise Exception("model_id(s) are required")

  count = int(kwargs['count']) if 'count' in kwargs and kwargs['count'] is not None else 1
  width = int(kwargs['width']) if 'width' in kwargs and kwargs['width'] is not None else 896
  height = int(kwargs['height']) if 'height' in kwargs and kwargs['height'] is not None else 640
  device_type = kwargs['device_type'] if 'device_type' in kwargs else get_device_type()
  model = make_one_pipeline(kwargs['model_id'], device_type)
  custom_latents = kwargs['custom_latents'] if 'custom_latents' in kwargs else True
  use_custom_latents = str(custom_latents).lower() != "false"
  seed = int(kwargs['seed']) if 'seed' in kwargs and kwargs['seed'] is not None else None
  input_paths = kwargs['input_paths'] if 'input_paths' in kwargs else None

  if input_paths == None and model['type'] != GENERATOR and 'previous_pass' in kwargs:
    input_paths = kwargs['previous_pass']['output_paths']

  seeds = []
  latents = []
  output_paths = []

  for _idx, path in enumerate(kwargs['output_paths'] if 'output_paths' in kwargs else make_empty_paths(count)):
    _latent_seed = None
    _latents = None
    path = maybe_update_path([
      add_name_to_path(model['short_name'])
    ])(path)

    if use_custom_latents == True:
      [_latent_seed, _latents] = make_latents(
        device_type,
        model['in_channels'],
        height=height,
        width=width,
        seed=seed,
      )
      seeds.append(_latent_seed)
      latents.append(_latents)
      output_paths.append(maybe_update_path([
        add_seed_to_path(str(_latent_seed))
      ])(path))
    else:
      seeds.append(None)
      latents.append(None)
      output_paths.append(maybe_update_path([
        add_seed_to_path("image")
      ])(path))

  return {
    "prompt": kwargs['prompt'],
    "negative_prompt": kwargs['negative_prompt'] if 'negative_prompt' in kwargs else None,
    "width": width,
    "height": height,
    "models": kwargs['models'],
    "model_id": kwargs['model_id'],
    "model": model,
    "latents": latents,
    "seeds": seeds,
    "steps": int(kwargs['steps'] if 'steps' in kwargs else 10),
    "input_paths": input_paths,
    "output_paths": output_paths,
    "device_type": device_type,
    "copyright": kwargs['copyright'] if 'copyright' in kwargs else 'losandes/diffusion-bench',
  }

def map_terminal_input (args):
  """
  Maps terminal input to args for img2img and txt2img
  """

  COPYRIGHT = args.copyright if args.copyright is not None else 'losandes/diffusion-bench'
  DEVICE = args.device_type if args.device_type is not None else get_device_type()
  PROMPT = split_to_list("|")(args.prompt)
  NEGATIVE_PROMPT = split_to_list("|")(args.negative_prompt)
  WIDTH = split_to_list(",")(args.width if args.width is not None else "896")
  HEIGHT = split_to_list(",")(args.height if args.height is not None else "640")
  COUNT = int(args.count if args.count is not None else 1)
  SEEDS = split_to_list(",")(args.seeds)
  MODELS = args.models if args.models is not None else "dreamlike-art/dreamlike-photoreal-2.0"
  MODEL_IDS = split_to_list(",")(MODELS)
  STEPS = split_to_list(",")(
    args.steps if args.steps is not None else "10"
  )
  INPUT_PATHS = split_to_list(",")(args.input_paths)
  OUTPUT_PATH = args.output_path if args.output_path is not None else "images"
  OUTPUT_PATH_TEMPLATE = args.output_path_template if args.output_path_template is not None else ":dir_path/:count_idx-:model_idx-:name-:seed.png"
  OUTPUT_PATHS = []
  CUSTOM_LATENTS = split_to_list(",")(args.custom_latents if args.custom_latents is not None else "True")

  for i in range(len(MODEL_IDS)):
    OUTPUT_PATHS.append(make_all_paths(OUTPUT_PATH_TEMPLATE)(OUTPUT_PATH, i, COUNT))

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
      copyright=COPYRIGHT,
      models=MODELS,
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
