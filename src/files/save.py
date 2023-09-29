import datetime
from PIL.ExifTags import TAGS
from ..options.split_to_list import split_to_list


def _stringifyModels(models, ensembleIdx):
  _models = []

  for idx, model in enumerate(split_to_list(",")(models)):
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

  exif[0x010E] = description     # https://www.awaresystems.be/imaging/tiff/tifftags/imagedescription.html
  exif[0x8298] = copyright       # https://www.awaresystems.be/imaging/tiff/tifftags/copyright.html
  exif[0x0131] = models[0]       # https://www.awaresystems.be/imaging/tiff/tifftags/software.html
  exif[0x9286] = comments        # https://www.awaresystems.be/imaging/tiff/tifftags/privateifd/exif/usercomment.html

  return exif

def save_image(image, path, prompt, seed, copyright, models, ensembleIdx):
  """
  Saves an image to disk and adds exif data

  Properties:
    image {Image} - the image to save
    path {string} - the location to save the image
    prompt {string} - the prompt that was used to generate the image
    copyright {string} - who this image's ownership should be attributed to
    models {string} - the list of models that was used to generate this image
    ensembleIdx {int} - the index of the model, among the models, that generated this image
  """
  _models = _stringifyModels(models, ensembleIdx)
  _comment = f"Trained algorithmic media using seed {seed}"

  image.save(path, exif=_appendExif(image, prompt, copyright, _models, _comment))

  return image
