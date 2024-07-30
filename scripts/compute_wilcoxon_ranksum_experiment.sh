#!/usr/bin/env bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

measure=$1  # dice
contrast=$2  # dwi

lut_fname=/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv

in_performance_fname_ref=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/dmri_hcp_t1/aggregate_performance/${measure}.tsv
ref_contrast=t1

# Build the labels for the i/o dirs/files
if [[ ${contrast} == "t1" ]]; then
  in_folder_label=dmri_hcp_t1
  file_basename_label=t1_resized
  out_folder_label=dmri_hcp_t1
elif [[ ${contrast} == "t2" ]]; then
  in_folder_label=dmri_hcp_t2
  file_basename_label=t2_resized
  out_folder_label=dmri_hcp_t2
elif [[ ${contrast} == "b0" ]]; then
  in_folder_label=dmri_hcp_b0
  file_basename_label=dwi_spherical_mean-b0_resized
  out_folder_label=dmri_hcp_b0
elif [[ ${contrast} == "dwi" ]]; then
  in_folder_label=dmri_hcp_sphm_b1000-2000-3000
  file_basename_label=dwi_spherical_mean-b1000-2000-3000_resized
  out_folder_label=dmri_hcp_sphm_b1000-2000-3000
elif [[ ${contrast} == "dwi1k" ]]; then
  in_folder_label=dmri_hcp_sphm_b1000
  file_basename_label=dwi_spherical_mean-b1000_resized
  out_folder_label=dmri_hcp_sphm_b1000
elif [[ ${contrast} == "dwi2k" ]]; then
  in_folder_label=dmri_hcp_sphm_b2000
  file_basename_label=dwi_spherical_mean-b2000_resized
  out_folder_label=dmri_hcp_sphm_b2000
elif [[ ${contrast} == "dwi3k" ]]; then
  in_folder_label=dmri_hcp_sphm_b3000
  file_basename_label=dwi_spherical_mean-b3000_resized
  out_folder_label=dmri_hcp_sphm_b3000
elif [[ ${contrast} == "fa" ]]; then
  in_folder_label=dmri_hcp_fa
  file_basename_label=fa_resized
  out_folder_label=dmri_hcp_fa
elif [[ ${contrast} == "md" ]]; then
  in_folder_label=dmri_hcp_md
  file_basename_label=md_resized
  out_folder_label=dmri_hcp_md
elif [[ ${contrast} == "rd" ]]; then
  in_folder_label=dmri_hcp_rd
  file_basename_label=rd_resized
  out_folder_label=dmri_hcp_rd
elif [[ ${contrast} == "evalse1" ]]; then
  in_folder_label=dmri_hcp_evals_e1
  file_basename_label=evals_e1_resized
  out_folder_label=dmri_hcp_evals_e1
elif [[ ${contrast} == "evalse2" ]]; then
  in_folder_label=dmri_hcp_evals_e2
  file_basename_label=evals_e2_resized
  out_folder_label=dmri_hcp_evals_e2
elif [[ ${contrast} == "evalse3" ]]; then
  in_folder_label=dmri_hcp_evals_e3
  file_basename_label=evals_e3_resized
  out_folder_label=dmri_hcp_evals_e3
elif [[ ${contrast} == "ak" ]]; then
  in_folder_label=dmri_hcp_ak
  file_basename_label=ak_resized
  out_folder_label=dmri_hcp_ak
elif [[ ${contrast} == "mk" ]]; then
  in_folder_label=dmri_hcp_mk
  file_basename_label=mk_resized
  out_folder_label=dmri_hcp_mk
elif [[ ${contrast} == "rk" ]]; then
  in_folder_label=dmri_hcp_rk
  file_basename_label=rk_resized
  out_folder_label=dmri_hcp_rk
else
  echo "Contrast not available:" ${contrast}
  echo "Aborting."
  exit 0
fi

in_performance_fname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/${in_folder_label}/aggregate_performance/${measure}.tsv

out_dirname=/mnt/data/cerebellum_parc/experiments_minimal_pipeline/${out_folder_label}/statistical_analysis
mkdir ${out_dirname}

python ~/src/dmriseg/scripts/compute_wilcoxon_ranksum.py \
  ${lut_fname} \
  ${measure} \
  /mnt/data/dmriseg/experiments/debugging/performance_cerebparc \
  --in_performance_fnames \
  ${in_performance_fname_ref} \
  ${in_performance_fname} \
  --in_contrast_names \
  ${ref_contrast} \
  ${contrast}
