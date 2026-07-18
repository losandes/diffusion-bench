import torch
from diffusers import HiDreamImagePipeline
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast


def makePipelines(device):
    """
    @see https://huggingface.co/HiDream-ai/HiDream-I1-Full
    txt-to-image

    NOTE: HiDream-I1 relies on a Llama-3.1 text encoder that is not bundled with
    the model repo, so it must be loaded separately and passed in as the fourth
    tokenizer/text encoder. meta-llama/Meta-Llama-3.1-8B-Instruct is gated; you
    must accept its license on HuggingFace and be authenticated to download it.
    """
    model_id = "HiDream-ai/HiDream-I1-Full"
    llama_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(llama_id)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        llama_id,
        output_hidden_states=True,
        output_attentions=True,
        torch_dtype=torch.bfloat16,
    )

    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        tokenizer_4=tokenizer_4,
        text_encoder_4=text_encoder_4,
        use_safetensors=True,
    ).to(device)

    # For M1/M2, enable memory efficient attention
    pipe.enable_attention_slicing()

    return [model_id, pipe]
