import torch
from diffusers import HiDreamImagePipeline


def makePipelines(device):
    """
    @see https://huggingface.co/HiDream-ai/HiDream-I1-Full
    txt-to-image
    """
    model_id = "HiDream-ai/HiDream-I1-Full"
    # tokenizer_4 = PreTrainedTokenizerFast.from_pretrained(
    #     "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # )
    # text_encoder_4 = LlamaForCausalLM.from_pretrained(
    #     "meta-llama/Meta-Llama-3.1-8B-Instruct",
    #     output_hidden_states=True,
    #     output_attentions=True,
    #     torch_dtype=torch.bfloat16,
    # )

    pipe = HiDreamImagePipeline.from_pretrained(
        model_id,
        # tokenizer_4=tokenizer_4,
        # text_encoder_4=text_encoder_4,
        # torch_dtype=torch.float16,  # Optimize for M1/M2
        use_safetensors=True,
        # variant="fp16",  # Use half precision variant if available
        requires_safety_checker=False,  # Don't require safety checker
    ).to(device)

    # For M1/M2, enable memory efficient attention
    pipe.enable_attention_slicing()

    return [model_id, pipe]
