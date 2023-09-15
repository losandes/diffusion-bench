from PIL import Image, ImageOps
import requests

def download_image(url):
  """
  Downloads an image from the internet and converts it
  to an RGB image for further processing
  """
  image = Image.open(requests.get(url, stream=True).raw)
  image = ImageOps.exif_transpose(image)
  image = image.convert("RGB")
  return image
