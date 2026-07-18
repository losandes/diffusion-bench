from diffusers import StableDiffusionPipeline


def makePipelines(device):
    """
    @see https://huggingface.co/HiDream-ai/HiDream-E1-Full
    any-to-any
    """
    model_id = "HiDream-ai/HiDream-E1-Full"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_safetensors=True,
    ).to(device)

    return [model_id, pipe]
