from diffusers import AutoPipelineForText2Image

def makePipelines (device):
  """
  @see https://huggingface.co/warp-ai/wuerstchen
  """
  model_id = "warp-diffusion/wuerstchen"
  pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    use_safetensors=True,
  ).to(device)

  return [model_id, pipe]
