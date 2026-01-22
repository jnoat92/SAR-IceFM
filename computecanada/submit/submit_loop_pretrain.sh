#!/bin/bash 
set -e
selfsup_configs=( 
# base
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/base/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py
# configs/selfsup/ai4arctic/pretrain_90/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt90.py
# configs/selfsup/ai4arctic/pretrain_95/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt95.py
# configs/selfsup/ai4arctic/pretrain_99/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt99.py

# large
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/large/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py
# configs/selfsup/ai4arctic/pretrain_90/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt90.py
# configs/selfsup/ai4arctic/pretrain_95/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt95.py
# configs/selfsup/ai4arctic/pretrain_99/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt99.py


# huge
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/huge/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py
# configs/selfsup/ai4arctic/pretrain_90/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt90.py
# configs/selfsup/ai4arctic/pretrain_95/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt95.py
# configs/selfsup/ai4arctic/pretrain_99/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt99.py

# SINGLE SCALE
configs/selfsup/ai4arctic_single_scale/pretrain_80/base/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80_sscale.py
# configs/selfsup/ai4arctic_single_scale/pretrain_40/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40_sscale.py

)

for i in "${!selfsup_configs[@]}"; do
   sbatch pretrain.sh ${selfsup_configs[i]}
   # echo  ${selfsup_configs[i]} $wandb_project
   echo "task successfully submitted" 
   sleep 2
done
