import torch

NONE = None

def make_latents (
  device,
  in_channels,
  width=768,
  height=768,
  seed=NONE,
):
  """
  @see https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb
  """
  # print(f"device: {device}")
  # print(f"in_channels: {in_channels}")
  # print(f"width: {width}")
  # print(f"height: {height}")
  # print(f"seed: {seed}")
  # print(f"together: {(1, in_channels, height // 8, width // 8)}")

  generator = torch.Generator(device=device)
  seed = seed if seed is not None else generator.seed()
  generator = generator.manual_seed(seed)

  return [
    seed,
    torch.randn(
      (1, in_channels, height // 8, width // 8),
      generator = generator,
      device = device
    ),
  ]

  # latents should have shape like (4, 4, 64, 64)
  # latents.shape


def _make_many_latents (
  device,
  in_channels,
  num_images=1,
  width=768,
  height=768,
  seed=NONE,
):
  """
  @see https://colab.research.google.com/github/pcuenca/diffusers-examples/blob/main/notebooks/stable-diffusion-seeds.ipynb
  """
  generator = torch.Generator(device=device)
  latents = None
  seeds = []

  for _ in range(num_images):
    # Get a new random seed, store it and use it as the generator state
    seed = seed if seed is not None else generator.seed()
    seeds.append(seed)
    generator = generator.manual_seed(seed)

    image_latents = torch.randn(
      # (1, pipe.unet.in_channels, h // 8, w // 8),
      (1, in_channels, height // 8, width // 8),
      generator = generator,
      device = device
    )
    latents = image_latents if latents is None else torch.cat((latents, image_latents))

  return [seeds, latents]

  # latents should have shape like (4, 4, 64, 64)
  # latents.shape
