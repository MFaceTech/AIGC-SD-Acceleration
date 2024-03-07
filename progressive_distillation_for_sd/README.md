## Progressive Distillation in Latent Space for Stable Diffusion
### Overview
The original Progressive Distillation is a technique developed by Google for use in Pixel Space. We have adapted it for application in Latent Space, specifically with Stable Diffusion models. This repository contains the reimplemented code, which is based on the diffusers library.

### Implementation Details
The provided example demonstrates how to distill the original 1000-step diffusion process down to 32 steps. Users can modify the code to further distill from 32 steps to 16, from 16 to 8, and so on, as needed.

### Usage
For the best results, we recommend using models with velocity-prediction. If you opt for models with epsilon-prediction, it is necessary to adjust the Signal-to-Noise Ratio (SNR) parameter accordingly.
```shell
export MODEL_PATH=
export DATASET_PATH=
export CACHE_PATH=
export OUTPUT_DIR=

accelerate launch --mixed_precision="fp16"  train_text_to_image_pd.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --dataset_name=$DATASET_PATH \
  --cache_dir=$CACHE_PATH \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=32 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=4000 \
  --num_steps_stu=32 \
  --num_steps_tea=1000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --output_dir=$OUTPUT_DIR
```
### Note
Please ensure that you have the **diffusers** library installed before running the code. The actual code for distillation will vary based on the specific implementation and parameters chosen.

### Acknowledgements
This work builds upon the innovative methods developed by Google and applies them to a new domain. We appreciate the contributions of the original authors and the open-source community.