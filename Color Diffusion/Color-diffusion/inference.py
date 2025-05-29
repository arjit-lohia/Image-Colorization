# import glob
# import torch
# import torchvision
# from dataset import ColorizationDataset, make_dataloaders
# from denoising import Unet, Encoder
# from utils import get_device, lab_to_rgb, load_default_configs, split_lab_channels
# from model import ColorDiffusion
# import numpy as np
# from argparse import ArgumentParser
# import os

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     checkpoints = glob.glob("C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/logs/training_logs/version_5/checkpoints/epoch=9-val_loss=0.0385.ckpt")
#     default_ckpt = checkpoints[-1] if checkpoints else None
#     # default_ckpt = "./checkpoints/last.ckpt"

#     parser.add_argument("-i", "--image-path", required=False, dest="img_path", default="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/test_images/bw3.jpg")
#     parser.add_argument("-T", "--diffusion-steps", default=350, dest="T")
#     # parser.add_argument("-T", "--diffusion-steps", default=500, dest="T")
#     parser.add_argument("--image-size", default=64, dest="img_size", type=int)
#     parser.add_argument("--checkpoint", default=default_ckpt, dest="ckpt")
#     parser.add_argument("--show", default=True)
#     parser.add_argument("--save", default=True)
#     parser.add_argument("--save_path", required=True, default="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/results/output1")
#     args = parser.parse_args()
#     assert args.ckpt is not None, "No checkpoint passed and ./checkpoints/ folder empty"

#     device = get_device()
#     enc_config, unet_config, colordiff_config = load_default_configs()
#     print("loaded default model config")
#     colordiff_config["T"] = args.T
#     colordiff_config["img_size"] = args.img_size

#     dataset = ColorizationDataset([args.img_path],
#                                   split="val",
#                                   config=colordiff_config)
#     image = dataset[0].unsqueeze(0)

#     encoder = Encoder(**enc_config)
#     unet = Unet(**unet_config)
#     model = ColorDiffusion.load_from_checkpoint(args.ckpt,
#                                                 strict=True,
#                                                 unet=unet,
#                                                 encoder=encoder,
#                                                 train_dl=None,
#                                                 val_dl=None,
#                                                 **colordiff_config)
#     model.to(device)

#     colorized = model.sample_plot_image(image.to(device),
#                                         show=args.show,
#                                         prog=True)
    
#     print(f"Type of colorized: {type(colorized)}")
#     # Move to the correct device
#     if isinstance(colorized, np.ndarray):
#         colorized = torch.tensor(colorized, dtype=torch.float32)  # Convert to PyTorch tensor

#     colorized = colorized.to(device)

#     print("Shape of colorized:", colorized.shape)

#     rgb_img = lab_to_rgb(*split_lab_channels(colorized))
    
#     # if args.save:
#     #     if args.save_path is None:
#     #         save_path = args.img_path + "colorized.jpg"
#     #     save_img = torch.tensor(rgb_img[0]).permute(2, 0, 1)
#     #     torchvision.utils.save_image(save_img, save_path)

#     if args.save:
#         os.makedirs(os.path.dirname(args.save_path), exist_ok=True)  # Ensure directory exists
#         save_img = torch.tensor(rgb_img[0]).permute(2, 0, 1)
#         torchvision.utils.save_image(save_img, args.save_path)
#         print(f"Image saved to {args.save_path}")











import glob
import torch
import torchvision
from dataset import ColorizationDataset, make_dataloaders
from denoising import Unet, Encoder
from utils import get_device, lab_to_rgb, load_default_configs, split_lab_channels
from model import ColorDiffusion
import numpy as np
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    checkpoints = glob.glob("C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/logs/training_logs/version_5/checkpoints/epoch=9-val_loss=0.0385.ckpt")
    default_ckpt = checkpoints[-1] if checkpoints else None
    # default_ckpt = "./checkpoints/last.ckpt"

    parser.add_argument("-i", "--image-path", required=False, dest="img_path", default="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/test_images/000007.jpg")
    parser.add_argument("-T", "--diffusion-steps", default=500, dest="T")
    # parser.add_argument("-T", "--diffusion-steps", default=500, dest="T")
    parser.add_argument("--image-size", default=128, dest="img_size", type=int)
    parser.add_argument("--checkpoint", default=default_ckpt, dest="ckpt")
    parser.add_argument("--show", default=True)
    parser.add_argument("--save", default=True)
    parser.add_argument("--save_path", required=True, default="C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/results/output1")
    args = parser.parse_args()
    assert args.ckpt is not None, "No checkpoint passed and ./checkpoints/ folder empty"

    device = get_device()
    enc_config, unet_config, colordiff_config = load_default_configs()
    print("loaded default model config")
    colordiff_config["T"] = args.T
    colordiff_config["img_size"] = args.img_size

    dataset = ColorizationDataset([args.img_path],
                                  split="val",
                                  config=colordiff_config)
    image = dataset[0].unsqueeze(0)

    encoder = Encoder(**enc_config)
    unet = Unet(**unet_config)
    model = ColorDiffusion.load_from_checkpoint(args.ckpt,
                                                strict=True,
                                                unet=unet,
                                                encoder=encoder,
                                                train_dl=None,
                                                val_dl=None,
                                                **colordiff_config)
    model.to(device)

    colorized = model.sample_plot_image(image.to(device),
                                        show=args.show,
                                        prog=True)
    
    print(f"Type of colorized: {type(colorized)}")
    # Move to the correct device
    if isinstance(colorized, np.ndarray):
        colorized = torch.tensor(colorized, dtype=torch.float32)  # Convert to PyTorch tensor

    colorized = colorized.squeeze(0).permute(2, 0, 1)  # [3, H, W]

    # Clip to [0,1] in case of any overflow
    colorized = colorized.clamp(0, 1)

    # Save directly
    if args.save:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        torchvision.utils.save_image(colorized, args.save_path)
        print(f"Image saved to {args.save_path}")

    # rgb_img = lab_to_rgb(*split_lab_channels(colorized))
    # rgb_img = (rgb_img * 255).clip(0, 255).astype(np.uint8)

    # if args.save:
    #     os.makedirs(os.path.dirname(args.save_path), exist_ok=True)  # Ensure directory exists
    #     save_img = torch.tensor(rgb_img[0]).permute(2, 0, 1)
    #     torchvision.utils.save_image(save_img, args.save_path)
    #     print(f"Image saved to {args.save_path}")
