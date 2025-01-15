import torch
import argparse
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from models import load_models
from utils import generate_image, save_image

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate images using Stable Diffusion.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save the generated image.")
    parser.add_argument("--batch", nargs="+", help="Batch of text prompts to generate multiple images.")
    args = parser.parse_args()

    # Load models
    vae, tokenizer, text_encoder, pipeline = load_models()

    # Example text prompt
    prompt = "Yosemite in winter"

    # Generate and save images
    if args.batch:
        print(f"Generating images for batch of prompts: {args.batch}")
        for i, prompt in enumerate(args.batch):
            image = generate_image(prompt, pipeline)
            image.show()
            output_path = f"{args.output.rsplit('.', 1)[0]}_{i}.png"
            save_image(image, output_path)
            print(f"Image for prompt '{prompt}' saved to {output_path}")
    else:
        print(f"Generating image for prompt: {args.prompt}")
        image = generate_image(args.prompt, pipeline)
        save_image(image, args.output)
        print(f"Image saved to {args.output}")

if __name__ == "__main__":
    main()

