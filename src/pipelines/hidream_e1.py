from diffusers import HiDreamImagePipeline


def makePipelines(device):
    """
    @see https://huggingface.co/HiDream-ai/HiDream-E1-Full
    any-to-any

    NOTE: HiDream-E1 is built on the HiDream-I1 architecture. The installed
    diffusers version does not ship a dedicated E1 editing pipeline, so we load
    it through the shared HiDreamImagePipeline family class.
    """
    model_id = "HiDream-ai/HiDream-E1-Full"
    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        use_safetensors=True,
    ).to(device)

    # For M1/M2, enable memory efficient attention
    pipe.enable_attention_slicing()

    return [model_id, pipe]
