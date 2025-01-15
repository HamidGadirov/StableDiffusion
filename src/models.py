import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline, AutoencoderKL

def load_models():
    """
    Load the Stable Diffusion pipeline and associated models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load Stable Diffusion pipeline
        pipeline = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
        )
        pipeline = pipeline.to(device)

        # Separate components
        vae = pipeline.vae
        tokenizer = pipeline.tokenizer
        text_encoder = pipeline.text_encoder

        print("Models loaded successfully.")
        return vae, tokenizer, text_encoder, pipeline

    except Exception as e:
        raise RuntimeError(f"Failed to load models: {e}")

