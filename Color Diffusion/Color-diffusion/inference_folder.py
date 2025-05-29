import glob
import torch
import torchvision
import os
from dataset import ColorizationDataset
from denoising import Unet, Encoder
from utils import get_device, load_default_configs
from model import ColorDiffusion
from PIL import Image
import numpy as np

def colorize_folder(input_folder, output_folder, checkpoint_path, T=200, img_size=64):
    os.makedirs(output_folder, exist_ok=True)
    
    device = get_device()
    enc_config, unet_config, colordiff_config = load_default_configs()
    colordiff_config["T"] = T
    colordiff_config["img_size"] = img_size

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    model = ColorDiffusion.load_from_checkpoint(
        checkpoint_path,
        strict=True,
        unet=unet,
        encoder=encoder,
        train_dl=None,
        val_dl=None,
        **colordiff_config
    )
    model.to(device)

    image_paths = sorted([
        os.path.join(input_folder, fname)
        for fname in os.listdir(input_folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for path in image_paths:
        dataset = ColorizationDataset([path], split="val", config=colordiff_config)
        image = dataset[0].unsqueeze(0).to(device)

        colorized = model.sample_plot_image(image, show=False, prog=True)

        if isinstance(colorized, np.ndarray):
            colorized = torch.tensor(colorized, dtype=torch.float32)

        colorized = colorized.squeeze(0).permute(2, 0, 1).clamp(0, 1)

        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, filename)
        torchvision.utils.save_image(colorized, save_path)
        print(f"Saved: {save_path}")

# Example usage
if __name__ == "__main__":
    colorize_folder(
        input_folder="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/test_images/100_bw",
        output_folder="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/results/100_output",
        checkpoint_path="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/logs/training_logs/version_5/checkpoints/epoch=9-val_loss=0.0385.ckpt"
    )