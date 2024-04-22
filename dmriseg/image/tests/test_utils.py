#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import disk

from dmriseg.image.utils import (
    one_hot2dist,  # , compute_distance_transform_map
)


def test_one_hot2dist():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    rx = 1
    ry = 1
    distmap = one_hot2dist(
        mask[np.newaxis], resolution=(rx, ry), dtype=np.float32
    )

    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
    cmap = "jet"
    im1 = ax[0].imshow(mask, cmap=cmap)
    ax1_divider = make_axes_locatable(ax[0])
    # Add an Axes to the right of the main Axes.
    cax1 = ax1_divider.append_axes("right", size="7%", pad="2%")
    fig.colorbar(im1, cax=cax1)
    ax[0].set_title("T")

    im2 = ax[1].imshow(distmap[0], cmap=cmap)
    # Add an Axes to the right of the main Axes.
    ax2_divider = make_axes_locatable(ax[1])
    cax2 = ax2_divider.append_axes("right", size="7%", pad="2%")
    fig.colorbar(im2, cax=cax2)
    ax[1].set_title("Distance transform map")
    plt.show()
