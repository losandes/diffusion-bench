import requests
from PIL import Image, ImageOps


def download_image(url):
    """
    Downloads an image from the internet and converts it
    to an RGB image for further processing
    """
    from io import BytesIO

    response = requests.get(url, stream=True)
    image = Image.open(BytesIO(response.content))
    if image is None:
        raise ValueError(f"Failed to load image from path: {url}")
    image = ImageOps.exif_transpose(image)
    if image is None:
        raise ValueError(f"Failed to transpose image exif: {url}")
    image = image.convert("RGB")
    return image
