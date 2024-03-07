# Acceleration for Stable Diffusion

## Overview
In this project, we provide a reproduction of acceleration schemes, such as Progressive Distillation[Progressive Distillation for Fast Sampling of Diffusion Models](https://arxiv.org/abs/2202.00512), MobileDiffusion[MobileDiffusion: Subsecond Text-to-Image Generation on Mobile Devices](https://arxiv.org/abs/2311.16567) and SDXL-Turbo[Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042), etc.

## 🔥 Update
- **[2024.3.6]** We release the first version of the code, containing progressive distillation, mobile diffusion unet and add turbo.

## Setup
- **Installation:**  
We recommend python version >= 3.8 and cuda version >= 11.4. And install the packages mentioned in the requirements.txt:
```bash
pip install -r requirements.txt
```

## Acknowledgements
This project build upon
- [diffusers](https://github.com/huggingface/diffusers)
- [stylegan-t](https://github.com/autonomousvision/stylegan-t)
- [distill-sd](https://github.com/segmind/distill-sd)
