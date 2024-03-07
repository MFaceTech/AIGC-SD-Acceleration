## ADD(Turbo) Training Code Reproduction
### Project Description
This repository contains our reproduced version of the ADD(Turbo) training code, which is based on the stylegan-t codebase.
The fundamental structure of the code is aligned with the descriptions provided in the original research paper.

### Development Status
Currently, the code is in an experimental phase. Users should be aware that there are known issues regarding the convergence
of the GAN (Generative Adversarial Network) component.

### Usage
#### Training
```shell
export MODEL_NAME="xxx"
export MODEL_PATH="xxx"
export CACHE_PATH="xxx"
export OUTPUT_PATH="xxx"

accelerate launch --mixed_precision="bf16" train_add.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir="" \
  --cache_dir=$CACHE_PATH \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --checkpointing_steps=500 \
  --max_train_steps=20000 \
  --snr_gamma=5.0 \
  --blur_fade_kimg=500 \
  --learning_rate_G=0.00001 \
  --learning_rate_D=0.00001 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$OUTPUT_PATH 


```

### Known Issues
Convergence: The GAN part of the code is experiencing convergence problems. We are actively working on debugging and 
refining the training process to achieve stable and reliable results.
### Usage Warning
Given that the code is experimental, it is recommended for research and development purposes only. We advise against using
this code for production environments until the convergence issues have been resolved.

### Contribution
We welcome contributions from the community to help resolve the current issues with the GAN convergence. Collaboration is key
to advancing the field, and we appreciate any insights or improvements that can be shared.

### Acknowledgments
Our work builds upon the innovative ideas presented in the ADD(Turbo) paper and the stylegan-t codebase. We acknowledge the original authors and contributors to these foundational resources.
