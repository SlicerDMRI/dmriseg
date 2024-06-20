#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

lr=2.0e-4
python /home/jhlegarreta/src/dmriseg/scripts/train_cerebparc.py \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/results/learning/segresnet16_batchsz1_lr2e4 \
  --lr ${lr}

lr=4.0e-4
python /home/jhlegarreta/src/dmriseg/scripts/train_cerebparc.py \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/results/learning/segresnet16_batchsz1_lr4e4 \
  --lr ${lr}

lr=6.0e-4
python /home/jhlegarreta/src/dmriseg/scripts/train_cerebparc.py \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/results/learning/segresnet16_batchsz1_lr6e4 \
  --lr ${lr}

lr=8.0e-4
python /home/jhlegarreta/src/dmriseg/scripts/train_cerebparc.py \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/train_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/img \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/valid_set/labelmaps \
  /mnt/data/cerebellum_parc/early_experiments/dmri_hcp_sphm_b1000-2000-3000_dwi_res_resized/results/learning/segresnet16_batchsz1_lr8e4 \
  --lr ${lr}
