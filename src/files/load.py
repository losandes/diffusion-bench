from PIL import Image, ImageOps

def load_image(path):
  """
  Loads an image from disk and converts it
  to an RGB image for further processing
  """
  image = Image.open(path)
  image = ImageOps.exif_transpose(image)
  return image.convert("RGB")
