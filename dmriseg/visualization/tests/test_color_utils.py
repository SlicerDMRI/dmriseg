#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest

from dmriseg.visualization.color_utils import (
    create_vtk_color_lut,
    get_or_create_cmap,
)
from dmriseg.visualization.plot_utils import show_mplt_colormap


@pytest.mark.skip(reason="Need to have testing data available.")
def test_get_or_create_cmap():

    cmap_name = "/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv"
    cmap = get_or_create_cmap(cmap_name)
    print(cmap)

    figsize = (15, 10)
    fig = show_mplt_colormap(cmap, figsize=figsize)
    fig.show()


@pytest.mark.skip(reason="Need to have testing data available.")
def test_create_vtk_color_lut():

    cmap_name = "/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv"
    _ = create_vtk_color_lut(cmap_name)
