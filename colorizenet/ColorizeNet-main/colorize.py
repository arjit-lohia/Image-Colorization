import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

# C:/Users/Arjit/OneDrive/Desktop/MTP/CODE/Color Diffusion/Color-diffusion/venv/Scripts/activate

from utils.data import HWC3, apply_color, resize_image
from utils.ddim import DDIMSampler
from utils.model import create_model, load_state_dict

model = create_model('./models/cldm_v21.yaml').cpu()
# model.load_state_dict(load_state_dict(
#     'lightning_logs/version_6/checkpoints/colorizenet-sd21.ckpt', location='cuda'))
model.load_state_dict(load_state_dict(
    'E:/MTP/colorizenet/ColorizeNet-main/logs/colorization/version_4/checkpoints/epoch=4-step=220796.ckpt', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)


input_image = cv2.imread("E:/MTP/colorizenet/ColorizeNet-main/NCD/Gray/Apple/Apple1.jpg")
# input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB) # comment
input_image = HWC3(input_image)
img = resize_image(input_image, resolution=512)
H, W, C = img.shape

num_samples = 1
control = torch.from_numpy(img.copy()).float().cuda() / 255.0
zeros = torch.zeros_like(control[:, :, :2])  # [H, W, 2]
control = torch.cat([control, zeros], dim=2)  # [H, W, 5]
control = torch.stack([control for _ in range(num_samples)], dim=0)
control = einops.rearrange(control, 'b h w c -> b c h w').clone()


# seed = random.randint(0, 65535)
seed = 1294574436
seed_everything(seed)
prompt = "Colorize this image"
n_prompt = ""
guess_mode = False
strength = 1.0
eta = 0.0
ddim_steps = 100
scale = 9.0

cond = {"c_concat": [control], "c_crossattn": [
    model.get_learned_conditioning([prompt] * num_samples)]}
un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [
    model.get_learned_conditioning([n_prompt] * num_samples)]}
shape = (4, H // 8, W // 8)

model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
    [strength] * 13)
samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                             shape, cond, verbose=False, eta=eta,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=un_cond)

x_samples = model.decode_first_stage(samples)
x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c')
             * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

results = [x_samples[i] for i in range(num_samples)]
colored_results = [apply_color(img, result) for result in results]

output_dir = "E:/MTP/colorizenet/ColorizeNet-main/NCD/output/Apple"
os.makedirs(output_dir, exist_ok=True)

[cv2.imwrite(os.path.join(output_dir,f"colorized_{i}.jpg"), cv2.cvtColor(result, cv2.COLOR_RGB2BGR)) for i, result in enumerate(colored_results)]