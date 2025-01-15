import torch
from PIL import Image

def generate_image(prompt, pipeline):
    """
    Generate an image based on the text prompt using Stable Diffusion.
    """
    with torch.no_grad():
        image = pipeline(prompt).images[0]
    return image

def pil_to_latent(vae, image, device):
    """
    Convert a PIL image to its latent representation.
    """
    from torchvision.transforms import ToTensor
    image_tensor = ToTensor()(image).unsqueeze(0).to(device) * 2 - 1
    with torch.no_grad():
        latents = 0.18215 * vae.encode(image_tensor).latent_dist.sample()
    return latents
