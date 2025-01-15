import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from models import load_models
from utils import generate_image

def main():
    # Load models
    vae, tokenizer, text_encoder, pipeline = load_models()

    # Example text prompt
    prompt = "Yosemite in winter"

    # Generate image
    image = generate_image(prompt, pipeline)
    image.show()

if __name__ == "__main__":
    main()
