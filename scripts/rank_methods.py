#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Rank the methods across metrics: get the performance scores average across
labels and compute the ranking for a given measure. Average the rank across
measures.
"""

import argparse
from pathlib import Path

import pandas as pd

from dmriseg.analysis.measures import Measure, is_ascending_sorting_better
from dmriseg.data.lut.utils import SuitAtlasDiedrichsenGroups
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import (
    append_label_to_fname,
    build_suffix,
    contrast_label,
    group_fname_label,
    stats_fname_label,
    underscore,
)
from dmriseg.stats.utils import mode_count
from dmriseg.utils.contrast_utils import (
    ContrastNames,
    get_dir_base_from_contrast_name,
)

performance_dir_label = "aggregate_performance"
file_basename = "group_measures_all_mean"

rank_fname_label = "rank"
avg_fname_label = "avg"
mean_fname_label = "mean"
measure_fname_label = "measure"
mode_fname_label = "mode"


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_root_dirname",
        help="Input dirname where performance data files dwell (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname (*.tsv)",
        type=Path,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    dvf_ext = DelimitedValuesFileExtension.TSV
    sep = "\t"

    data_suffix = build_suffix(dvf_ext)

    measures = [
        Measure.DICE.value,
        Measure.HAUSDORFF95.value,
        Measure.MEAN_SURFACE_DISTANCE.value,
        Measure.CENTER_OF_MASS_DISTANCE.value,
        Measure.VOLUME_SIMILARITY.value,
        Measure.LABEL_DETECTION_RATE.value,
    ]

    contrasts = [
        ContrastNames.T1.value,
        ContrastNames.B0.value,
        ContrastNames.DWI.value,
        ContrastNames.DWI1k.value,
        ContrastNames.DWI2k.value,
        ContrastNames.DWI3k.value,
        ContrastNames.FA.value,
        ContrastNames.MD.value,
        ContrastNames.RD.value,
        ContrastNames.EVALS_E1.value,
        ContrastNames.EVALS_E2.value,
        ContrastNames.EVALS_E3.value,
        ContrastNames.AK.value,
        ContrastNames.MK.value,
        ContrastNames.RK.value,
    ]

    measures_df = pd.DataFrame(index=contrasts, columns=measures)
    measures_df.index.name = contrast_label

    rank_df = pd.DataFrame(index=contrasts, columns=measures)
    rank_df.index.name = contrast_label

    # Loop over measures
    for measure in measures:

        # Loop over contrasts
        mean_scores = dict({})
        for contrast in contrasts:
            dir_label = get_dir_base_from_contrast_name(contrast)
            dirname = args.in_root_dirname / dir_label / performance_dir_label
            file_basename = (
                group_fname_label
                + underscore
                + measure
                + underscore
                + stats_fname_label
                + data_suffix
            )
            fname = dirname / file_basename
            df = pd.read_csv(fname, sep=sep, index_col=0)
            mean_score = df.loc["mean"][SuitAtlasDiedrichsenGroups.ALL.value]
            mean_scores[contrast] = mean_score

        measures_df[measure] = mean_scores

    # Rank each column in according to the best sorting for the measure (e.g.
    # descending for dice (higher is better), ascending for hd95 (lower is
    # better), etc.)
    ranking_order = {
        measure: is_ascending_sorting_better(measure) for measure in measures
    }
    for col, order in ranking_order.items():
        rank_df[col] = (
            measures_df[col].rank(ascending=order, method="dense").astype(int)
        )

    # Average across columns
    avg_rank_df = rank_df.mean(axis=1)
    avg_rank_df.index.name = contrast_label

    # Compute the mode and the corresponding value on each row
    # Apply the function to each row
    mode_df = rank_df.apply(mode_count, axis=1)
    mode_df.index.name = contrast_label

    # Save the data
    file_basename = (
        group_fname_label
        + underscore
        + measure_fname_label
        + underscore
        + SuitAtlasDiedrichsenGroups.ALL.value
        + underscore
        + mean_fname_label
        + data_suffix
    )
    fname = args.out_dirname / file_basename
    measures_df.to_csv(fname, sep=sep)

    fname = append_label_to_fname(fname, rank_fname_label)
    rank_df.to_csv(fname, sep=sep)

    fname = append_label_to_fname(fname, avg_fname_label)
    avg_rank_df.to_csv(fname, sep=sep)

    fname = append_label_to_fname(fname, mode_fname_label)
    mode_df.to_csv(fname, sep=sep)


if __name__ == "__main__":
    main()
