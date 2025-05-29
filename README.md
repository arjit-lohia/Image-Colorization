# Black and White Image Colorization - Four Approaches

This repository showcases four different methods for colorizing black and white images. Each approach leverages a different deep learning technique or model architecture.

---

## 1. Color Diffusion

A simple diffusion-based colorization approach trained on the CelebA dataset.

- Works in LAB color space.
  ![LAB Color Space](IMAGES/lab_color_space.png)
- Greyscale "L" channel is given to the model as input.
- UNet predicts noise on AB channels during training.
- Conditional encoder extracts intermediate features from the L channel.

![Forward Diffusion](https://github.com/ErwannMillon/Color-diffusion/blob/main/visualization/forward_diff.gif)

![Denoising](https://github.com/ErwannMillon/Color-diffusion/blob/main/visualization/train/total1.gif)

![Colorization Example](IMAGES/DDPM_output.png)

---

## 2. ColorizeNet (ControlNet with Stable Diffusion v2.1)

A ControlNet-based image colorization model trained on grayscale COCO images using prompts generated from InstructBLIP.

- Finetunes Stable Diffusion v2.1.
- Requires grayscale-color pairs and prompts in JSON format.
- Supports training, inference, and downloading pretrained weights.

![Colorization Example](IMAGES/SD_output.png)

[ControlNet Repo](https://github.com/lllyasviel/ControlNet)

---

## 3. CT² - Colorization Transformer via Color Tokens

Transformer-based colorization that tackles undersaturation and semantic ambiguity.

- Uses ViT as the backbone.
- Color tokens introduced into training.
- Trained and tested on ImageNet.
- Distributed training supported.

![Colorization Example](IMAGES/CT2.png)

[Citation Paper](https://ci.idm.pku.edu.cn/Weng_ECCV22b.pdf)

---

## 4. Text-Guided Image Colorization

Interactive colorization using SDXL or SDXL-Lightning with ControlNet and BLIP captions.

- Allows users to control object colors via prompts.
- Gradio UI available for demo.
- SDXL-based training and evaluation scripts provided.

---
