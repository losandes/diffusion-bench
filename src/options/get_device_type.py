import torch, sys, platform

def get_device_type ():
  """
  Returns the type of device to port pipelines to
  (e.g. cuda, cpu, or mps)
  """
  if torch.cuda.is_available():
    # nvidia GPU
    return "cuda"
  elif sys.platform == "darwin" and platform.processor() == "arm":
    # Macbook Pro M1/M2:
    return "mps"
  else:
    return "cpu"
