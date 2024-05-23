#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import nibabel as nib
import pytest
from fury import window
from matplotlib import pyplot as plt
from PIL import Image

from dmriseg.anatomy.utils import Axis
from dmriseg.image.utils import (
    create_mask_from_label_image,
    extract_roi_from_label_image,
)
from dmriseg.visualization.color_utils import create_vtk_color_lut
from dmriseg.visualization.scene_utils import (
    contour_actor_kwargs_name,
    create_image_from_scene,
    create_slice_roi_scene,
    rgb2gray4pil,
    screenshot_slice,
)


@pytest.mark.skip(reason="Need to have testing data available.")
def test_create_slice_roi_scene():

    # Load a SUIT segmentation
    fname = "/mnt/data/connectome/suit/results/cer_seg_100307.nii"
    label_img = nib.load(fname)

    # Extract ROI using the label of interest
    label = 8
    _label_img = extract_roi_from_label_image(label_img, label)

    # Create a binary image
    img_data = create_mask_from_label_image(_label_img)
    roi_img = nib.nifti1.Nifti1Image(img_data, affine=label_img.affine)

    # Load the corresponding T1
    fname = "/mnt/data/connectome/s1200_download_on_20170516/downloaded/3T_structural_preproc/100307/T1w/T1w_acpc_dc_restore_brain.nii.gz"
    structural_img = nib.load(fname)

    # cmap = get_atlas_cmap(Atlas.DKT)
    import numpy as np

    color = [
        np.array([1, 0, 0]),
    ]
    opacity = [
        1.0,
    ]
    contour_actor_kwargs = dict({"color": color, "opacity": opacity})
    kwargs = dict({contour_actor_kwargs_name: contour_actor_kwargs})

    axis = Axis.AXIAL
    slice_idx = 66
    scene = create_slice_roi_scene(
        structural_img, [roi_img], axis, slice_idx, **kwargs
    )

    size = (1024, 720)
    reset_camera = False
    showm = window.ShowManager(scene, size=size, reset_camera=reset_camera)
    showm.initialize()
    showm.start()


@pytest.mark.skip(reason="Need to have testing data available.")
@pytest.mark.parametrize(
    ("fname", "axis_name", "cmap_name"),
    [
        (
            "/mnt/data/test_data/tractodata/datasets/hcp_ya/100307/100307wmparc_reshape_T1w_acpc_dc_restore_1.25.nii.gz",
            "axial",
            "viridis",
        ),
        (
            "/mnt/data/test_data/tractodata/datasets/hcp_ya/100307/100307_cer_seg.nii.gz",
            "sagittal",
            "/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv",
        ),
    ],
)
def test_screenshot_slice(fname, axis_name, cmap_name):

    img = nib.load(fname)
    axis = Axis(axis_name)
    slice_ids = [70]
    size = (768, 768)

    lookup_colormap = None
    if Path(cmap_name).is_file():
        lookup_colormap = create_vtk_color_lut(cmap_name)

    kwargs = dict(
        {"interpolation": "nearest", "lookup_colormap": lookup_colormap}
    )
    scene_container = screenshot_slice(img, axis, slice_ids, size, **kwargs)

    if Path(cmap_name).is_file():
        labelmap_arr = scene_container[0]
        image = Image.fromarray(labelmap_arr, mode="RGB")
    else:
        labelmap_arr = rgb2gray4pil(scene_container[0])
        cmap = plt.get_cmap(cmap_name)
        # data returned by cmap is normalized to the [0,1] range: scale to the
        # [0, 255] range and convert to uint8 for Pillow
        _arr = (cmap(labelmap_arr) * 255).astype("uint8")
        image = Image.fromarray(_arr, mode=None)

    image.show()


@pytest.mark.skip(reason="Need to have testing data available.")
@pytest.mark.parametrize(
    ("fname", "axis_name", "cmap_name"),
    [
        (
            "/mnt/data/test_data/tractodata/datasets/hcp_ya/100307/100307wmparc_reshape_T1w_acpc_dc_restore_1.25.nii.gz",
            "axial",
            "viridis",
        ),
        (
            "/mnt/data/test_data/tractodata/datasets/hcp_ya/100307/100307_cer_seg.nii.gz",
            "sagittal",
            "/mnt/data/lut/suit_diedrichsen_lut0255_nuclei_colored.tsv",
        ),
    ],
)
def test_create_image_from_scene(fname, axis_name, cmap_name):

    img = nib.load(fname)
    axis = Axis(axis_name)
    slice_ids = [70]
    size = (768, 768)

    lookup_colormap = None
    if Path(cmap_name).is_file():
        lookup_colormap = create_vtk_color_lut(cmap_name)

    kwargs = dict(
        {"interpolation": "nearest", "lookup_colormap": lookup_colormap}
    )
    scene_container = screenshot_slice(img, axis, slice_ids, size, **kwargs)

    if Path(cmap_name).is_file():
        cmap_name = None
        _labelmap_arr = scene_container[0]
    else:
        _labelmap_arr = rgb2gray4pil(scene_container[0])

    labelmap_img = create_image_from_scene(
        _labelmap_arr, size, cmap_name=cmap_name
    )
    labelmap_img.show()
