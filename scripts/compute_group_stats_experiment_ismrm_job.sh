#!/bin/bash

# Source the virtualenvwrapper.sh script
source $(which virtualenvwrapper.sh)

workon dmriseg

in_contrast_names=("t1" "b0" "dwi" "dwi1k" "dwi2k" "dwi3k")

echo "Computing group stats..."
for _contrast_name in "${in_contrast_names[@]}"; do

  echo ${_contrast_name}

  ~/src/dmriseg/scripts/compute_group_stats_experiment_ismrm.sh ${_contrast_name}

done

echo "Finished computing group stats."
