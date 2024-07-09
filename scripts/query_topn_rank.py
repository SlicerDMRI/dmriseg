#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Query the top N performance data for a given measure name. If no N is specified,
all participants are returned. If no group name is specified, participants will
be ranked according to the average performance across all labels.
Useful to extract the participant outperforming the rest so that their results
are used in plots.
"""

import argparse
from pathlib import Path

import pandas as pd

from dmriseg.analysis.measures import Measure
from dmriseg.data.lut.utils import (
    SuitAtlasDiedrichsenGroups,
    get_diedrichsen_group_labels,
)
from dmriseg.io.file_extensions import DelimitedValuesFileExtension
from dmriseg.io.utils import build_suffix, participant_label_id, underscore

topn_label = "top"


def compute_group_performance(df, group_name):

    # Select the labels corresponding to the group of interest and compute stats
    # across all elements
    labels = list(map(str, get_diedrichsen_group_labels(group_name)))
    df_subset = df[labels]

    with pd.option_context("mode.use_inf_as_na", True):
        df_group_mean = pd.DataFrame(
            df_subset.mean(axis=1).values, columns=[group_name], index=df.index
        )
        return df_group_mean


def _build_arg_parser():

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "in_performance_dirname",
        help="Input dirname where performance data files dwell (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "out_dirname",
        help="Output dirname (*.tsv)",
        type=Path,
    )
    parser.add_argument(
        "measure_name",
        help="Measure name",
        type=str,
    )
    parser.add_argument(
        "--top_n",
        help="Number of top scoring participants. If not given, all "
        "participants will be returned.",
        type=int,
    )
    parser.add_argument(
        "--rank_group",
        help="Group name for ranking purposes. Only used if top_n is given.",
        type=str,
    )
    return parser


def _parse_args(parser):

    args = parser.parse_args()

    return args


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    ext = DelimitedValuesFileExtension.TSV
    sep = "\t"

    measure = Measure(args.measure_name).value

    suffix = build_suffix(ext)
    file_basename = measure + suffix
    fname = args.in_performance_dirname / file_basename

    df = pd.read_csv(fname, sep=sep, index_col=participant_label_id)

    group_names = [
        SuitAtlasDiedrichsenGroups.ALL,
        SuitAtlasDiedrichsenGroups.DCN,
        SuitAtlasDiedrichsenGroups.DENTATE,
        SuitAtlasDiedrichsenGroups.INTERPOSED,
        SuitAtlasDiedrichsenGroups.FASTIGIAL,
        SuitAtlasDiedrichsenGroups.VERMIS,
        SuitAtlasDiedrichsenGroups.LOBULES,
        SuitAtlasDiedrichsenGroups.CRUS,
    ]

    # Put inside a method query_top_scores

    mean_df = pd.DataFrame()
    for group_name in group_names:
        # Compute the mean across all labels in group name
        _mean_df = compute_group_performance(df, group_name.value)

        # Concatenate to existing df horizontally
        mean_df = pd.concat([mean_df, _mean_df], axis=1)

    top_n = args.top_n
    if top_n is None:
        top_n = len(mean_df)

    rank_group = SuitAtlasDiedrichsenGroups(args.rank_group).value
    if rank_group is None:
        rank_group = SuitAtlasDiedrichsenGroups.ALL.value

    topn_rank_df = mean_df.sort_values([rank_group], ascending=False).head(
        top_n
    )

    suffix = build_suffix(ext)
    file_basename = (
        measure
        + underscore
        + rank_group
        + underscore
        + topn_label
        + str(top_n)
        + suffix
    )
    out_fname = args.out_dirname / file_basename
    topn_rank_df.to_csv(out_fname, sep=sep)


if __name__ == "__main__":
    main()
