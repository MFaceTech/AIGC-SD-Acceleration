## MobileDiffusion UNET Reproduction

### Overview

Our team has successfully replicated the UNet compression and acceleration aspects of MobileDiffusion, 
which includes the pruning and merging of the down and up layers, as well as the weight sharing in the attention mechanism.

### Usage and Application
```shell
export MODEL_PATH="XXX"
export DATASET_PATH="XXX"

accelerate launch --mixed_precision="fp16"  distill_training_dalcefo_md.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --train_data_dir=$DATASET_PATH \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=16 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=25000 \
  --distill_level="md"\
  --prepare_unet="True" \
  --output_weight=10 \
  --feature_weight=0.5 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir="XXX" \
  --student_unet_weight="XXX"

```

### Contribution and Collaboration

We are open to contributions from the community to refine and improve upon the current implementation. 

### Acknowledgments

We would like to acknowledge the original authors of MobileDiffusion for their innovative approach to model
compression and acceleration. Our work is a testament to the impact of their research in the field of 
efficient machine learning models.
