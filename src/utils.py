import torch
from PIL import Image

def generate_image(prompt, pipeline):
    """
    Generate an image based on the text prompt using Stable Diffusion.
    """
    if not prompt or not isinstance(prompt, str):
        raise ValueError("Please provide a valid text prompt.")
    
    print(f"Generating image for prompt: '{prompt}'")
    with torch.no_grad():
        image = pipeline(prompt).images[0]
    return image

def save_image(image, output_path):
    """
    Save a PIL image to the specified file path.
    """
    try:
        image.save(output_path)
        print(f"Image saved at {output_path}")
    except Exception as e:
        print(f"Failed to save image: {e}")

def pil_to_latent(vae, image, device):
    """
    Convert a PIL image to its latent representation using the VAE.
    """
    from torchvision.transforms import ToTensor
    image_tensor = ToTensor()(image).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        latents = 0.18215 * vae.encode(image_tensor).latent_dist.sample()
    return latents

