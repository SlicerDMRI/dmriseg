#!/usr/bin/env bash

dirname="/mnt/data/cerebellum_parc/experiments_minimal_pipeline/plots/figures/performance_plots_aggr"

file_rootnames=(
  "dice_all"
  "hd95_all"
  "msd_all"
  "cm_dist_all"
  "vs_all"
  "lab_detect_rate_all"
  "dice_anterior_lobe"
  "hd95_anterior_lobe"
  "msd_anterior_lobe"
  "cm_dist_anterior_lobe"
  "vs_anterior_lobe"
  "lab_detect_rate_anterior_lobe"
  "dice_posterior_lobe"
  "hd95_posterior_lobe"
  "msd_posterior_lobe"
  "cm_dist_posterior_lobe"
  "vs_posterior_lobe"
  "lab_detect_rate_posterior_lobe"
)

export_dpi=300

# Convert SVG files to PNG
for _file_rootname in "${file_rootnames[@]}"; do

  echo "File rootname:" ${_file_rootname}

  in_fname=${dirname}/${_file_rootname}.svg
  out_fname=${dirname}/${_file_rootname}.png

  inkscape ${in_fname} --export-type=png --export-filename=${out_fname} --export-dpi=${export_dpi}

done

# Convert the legend figure
in_fname=${dirname}/"dice_all_legend.svg"
out_fname=${dirname}/"all_legend.png"

inkscape ${in_fname} --export-type=png --export-filename=${out_fname} --export-dpi=${export_dpi}

# Stack the plots and legends vertically
for _file_rootname in "${file_rootnames[@]}"; do

  echo "File rootname:" ${_file_rootname}

  in_fname1=${dirname}/${_file_rootname}.png
  in_fname2=${dirname}/"all_legend.png"
  out_fname=${dirname}/${_file_rootname}_legend.png

  # Resize the legend to the size of the plot
  dim=$(identify -format "%wx%h" ${in_fname1})

  in_fname2_resized=${dirname}/"all_legend_resized.png"

  convert ${in_fname2} -resize ${dim} ${in_fname2_resized}

  convert ${in_fname1} ${in_fname2_resized} -append ${out_fname}

done
