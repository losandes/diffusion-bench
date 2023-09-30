from ..options.split_to_list import split_to_list


def _stringifyModels(models, ensembleIdx):
  _models = []

  for _idx, model in enumerate(split_to_list(",")(models)):
    _models.append(f"\"{model}\"")

  _joined = ", ".join(_models)

  return [f"[{_joined}][{ensembleIdx}]", _models[ensembleIdx]]

def _appendExif (image, description, copyright, models, comments):
  """
  Gets the existing exif tags and appends / overwrites them

  @see https://pillow.readthedocs.io/en/stable/reference/ExifTags.html
  @see https://www.awaresystems.be/imaging/tiff/tifftags/baseline.html
  @see https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif.html
  """
  exif = image.getexif()

  exif[0x010E] = description     # ImageDescription: https://www.awaresystems.be/imaging/tiff/tifftags/imagedescription.html
  exif[0x8298] = copyright       # Copyright: https://www.awaresystems.be/imaging/tiff/tifftags/copyright.html
  exif[0x0131] = models[0]       # Software: https://www.awaresystems.be/imaging/tiff/tifftags/software.html
  exif[0x9286] = comments        # UserComment: https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif/usercomment.html

  return exif

def save_image(image, path, seed, ensembleIdx, options):
  """
  Saves an image to disk and adds exif data

  Properties:
    image (Image) - the image to save
    path (string) - the location to save the image
    seed (string) - the PRNG seed that was used to produce this image
    ensembleIdx (int) - the index of the model, among the models, that generated this image
    options (dict) - the options that were passed to the generator
  """
  prompt = options['prompt']
  negative_prompt = options['negative_prompt']
  steps = options['steps']
  copyright = options['copyright']
  models = options['models']

  _models = _stringifyModels(models, ensembleIdx)
  _comment = f"Trained algorithmic media, seed {seed}, steps: {steps}, negative-prompt: {negative_prompt}"

  image.save(path, exif=_appendExif(image, prompt, copyright, _models, _comment))
  image.close()
