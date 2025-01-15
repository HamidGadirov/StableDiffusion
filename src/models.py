import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL

def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        revision="fp16",
        torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)

    # Separate components
    vae = pipeline.vae
    tokenizer = pipeline.tokenizer
    text_encoder = pipeline.text_encoder

    return vae, tokenizer, text_encoder, pipeline
