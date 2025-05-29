from torchinfo import summary

from denoising import Unet, Encoder
from utils import load_default_configs

enc_config, unet_config, colordiff_config = load_default_configs()

encoder = Encoder(**enc_config)
unet = Unet(**unet_config)

# print(unet)

summary(encoder)