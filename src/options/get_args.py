import argparse
from constants import GENERATOR
from options.get_device_type import get_device_type
from options.list_to_ints import list_to_ints
from options.make_paths import make_paths
from options.split_to_list import split_to_list
from pipelines.make_pipelines import make_pipelines

def get_args ():
  """
  Gets the args that were passed with the terminal command and
  returns them in a dict with defaults for missing args
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
  args = parser.parse_args()

  DEVICE = args.device_type if args.device_type is not None else get_device_type()
  PROMPT = split_to_list("|")(args.prompt)
  NEGATIVE_PROMPT = split_to_list("|")(args.negative_prompt)
  WIDTH = int(args.width if args.width is not None else 896)
  HEIGHT = int(args.height if args.height is not None else 640)
  COUNT = int(args.count if args.count is not None else 1)
  SEEDS = list_to_ints(split_to_list(",")(args.seeds))
  MODEL_IDS = split_to_list(",")(
    args.models if args.models is not None
    else "dreamlike-art/dreamlike-photoreal-2.0, stabilityai/stable-diffusion-xl-refiner-1.0, stabilityai/stable-diffusion-xl-refiner-1.0, stabilityai/stable-diffusion-xl-refiner-1.0"
  )
  STEPS = list_to_ints(split_to_list(",")(
    args.steps if args.steps is not None else "10, 50, 50, 50"
  ))
  INPUT_PATHS = split_to_list(",")(args.input_paths)
  OUTPUT_PATH = args.output_path if args.output_path is not None else "images"
  OUTPUT_PATH_TEMPLATE = args.output_path_template if args.output_path_template is not None else ":path/:count_idx-:type-:model_idx.png"
  PATHS = []
  CUSTOM_LATENTS = True
  MODELS = make_pipelines(MODEL_IDS, DEVICE)

  for i in range(len(MODEL_IDS)):
    PATHS.append(make_paths(OUTPUT_PATH_TEMPLATE)(OUTPUT_PATH, MODELS[i][0], i, COUNT))

  if MODELS[0][0] is not GENERATOR and len(INPUT_PATHS) == 0:
    raise Exception("Input paths (-i or --input_paths) are required when the first model is not a generator")

  if PROMPT is None:
    raise Exception("A prompt (-p or --prompt) is required")

  if len(MODELS) != len(STEPS):
    raise Exception("The models (-m or --models) and steps (-s or --steps) lists must be of the same length")

  print("")
  print("====================================================")
  print(f"prompt:          {PROMPT}")
  print(f"negative_prompt: {NEGATIVE_PROMPT}")
  print(f"width:           {WIDTH}")
  print(f"height:          {HEIGHT}")
  print(f"count:           {COUNT}")
  print(f"seeds:           {SEEDS}")
  print(f"model_ids:       {MODEL_IDS}")
  print(f"steps:           {STEPS}")
  print(f"input_paths:     {INPUT_PATHS}")
  print(f"output_paths:    {PATHS}")
  print(f"device:          {DEVICE}")
  print("====================================================")
  print("")

  return {
    "PROMPT": PROMPT,
    "NEGATIVE_PROMPT": NEGATIVE_PROMPT,
    "WIDTH": WIDTH,
    "HEIGHT": HEIGHT,
    "COUNT": COUNT,
    "SEEDS": SEEDS,
    "MODEL_IDS": MODEL_IDS,
    "MODELS": MODELS,
    "STEPS": STEPS,
    "INPUT_PATHS": INPUT_PATHS,
    "PATHS": PATHS,
    "DEVICE": DEVICE,
    "CUSTOM_LATENTS": CUSTOM_LATENTS,
  }
