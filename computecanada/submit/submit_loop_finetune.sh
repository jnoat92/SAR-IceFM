#!/bin/bash 
set -e
selfsup_configs=( 
# # base
configs/selfsup/ai4arctic/pretrain_20/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/base/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py
# configs/selfsup/ai4arctic/pretrain_90/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt90.py
# configs/selfsup/ai4arctic/pretrain_95/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt95.py
# configs/selfsup/ai4arctic/pretrain_99/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt99.py

# # large
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/large/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py
# configs/selfsup/ai4arctic/pretrain_90/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt90.py
# configs/selfsup/ai4arctic/pretrain_95/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt95.py
# configs/selfsup/ai4arctic/pretrain_99/mae_vit-large-p16_4xb8-amp-coslr-30ki_ai4arctic_pt99.py


# # # huge
# configs/selfsup/ai4arctic/pretrain_20/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt20.py
# configs/selfsup/ai4arctic/pretrain_40/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40.py
# configs/selfsup/ai4arctic/pretrain_60/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt60.py
# configs/selfsup/ai4arctic/pretrain_80/huge/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80.py
# configs/selfsup/ai4arctic/pretrain_90/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt90.py
# configs/selfsup/ai4arctic/pretrain_95/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt95.py
# configs/selfsup/ai4arctic/pretrain_99/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt99.py

# SINGLE SCALE
# configs/selfsup/ai4arctic_single_scale/pretrain_80/base/mae_vit-base-p16_4xb8-amp-coslr-30ki_ai4arctic_pt80_sscale.py
# configs/selfsup/ai4arctic_single_scale/pretrain_40/mae_vit-huge-p16_4xb8-amp-coslr-30ki_ai4arctic_pt40_sscale.py

)

fintune_configs=( 
# # base
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft40.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft10.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft05.py
# configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft01.py

# # large
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft40.py   
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft10.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft05.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft01.py

# # huge
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft80.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft40.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft10.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft05.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft01.py

# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr60.py
# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr65.py
# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr70.py
# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr75.py
# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr80.py
# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr85.py
# # # configs/multi_task_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr90.py

# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr60.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr65.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr70.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr75.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr80.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr85.py
# configs/multi_task_ai4arctic/vit/mae_vit-large_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr90.py

# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr60.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr65.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr70.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr75.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr80.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr85.py
# configs/multi_task_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft20_mr90.py

# ========== SINGLE TASK
# # SIC
# configs/single_task_ai4arctic/vit/SIC/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_SIC.py
# configs/single_task_ai4arctic/vit/SIC/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60_SIC.py
# # SOD
# configs/single_task_ai4arctic/vit/SOD/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_SOD.py
# configs/single_task_ai4arctic/vit/SOD/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60_SOD.py
# # FLOE
# configs/single_task_ai4arctic/vit/FLOE/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_FLOE.py
# configs/single_task_ai4arctic/vit/FLOE/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60_FLOE.py

# # ========== SINGLE SCALE
# configs/single_scale_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_sscale.py
# configs/single_scale_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60_sscale.py

# configs/single_scale_ai4arctic/vit/mae_vit-base_4xb8-amp-coslr-30ki_ai4arctic_ft20_sscale_fmulti.py
configs/single_scale_ai4arctic/vit/mae_vit-huge_4xb8-amp-coslr-30ki_ai4arctic_ft60_sscale_fmulti.py
)


for i in "${!selfsup_configs[@]}"; do
   sbatch finetune.sh ${selfsup_configs[i]} ${fintune_configs[i]}
   echo "task successfully submitted" 
   sleep 2
done

