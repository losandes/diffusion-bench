from PIL import Image, ImageOps


def load_image(path):
    """
    Loads an image from disk and converts it
    to an RGB image for further processing
    """
    print(f" LOADINGPATH {path}")
    image = Image.open(path)
    if image is None:
        raise ValueError(f"Failed to load image from path: {path}")
    image = ImageOps.exif_transpose(image)
    if image is None:
        raise ValueError(f"Failed to transpose image exif: {path}")
    return image.convert("RGB")
