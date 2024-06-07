#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import nibabel as nib

from dmriseg.anatomy.utils import Axis

# from dmriseg.data.lut.utils import Atlas, fetch_atlas_cmap
from dmriseg.image.utils import assert_same_resolution  # , check_slice_indices
from dmriseg.visualization.color_utils import create_vtk_color_lut
from dmriseg.visualization.scene_utils import (  # check_mosaic_layout,
    compose_mosaic,
    screenshot_slice,
)

# from dmriseg.io.utils import assert_inputs_exist, assert_outputs_exist, ranged_type


def _build_arg_parser():

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )

    # Positional arguments
    p.add_argument("in_vol", help="Input volume image file.")
    p.add_argument("in_transparency_mask", help="Input mask image file.")
    p.add_argument(
        "out_fname",
        help="Name of the output image mosaic (e.g. mosaic.jpg, mosaic.png).",
    )
    p.add_argument(
        "slice_ids", nargs="+", type=int, help="Slice indices for the mosaic."
    )
    p.add_argument(
        "mosaic_rows_cols",
        nargs=2,
        # metavar=("rows", "cols"),  # CPython issue 58282
        type=int,
        help="The mosaic row and column count.",
    )

    # Optional arguments
    p.add_argument("--in_labelmap", help="Labelmap image.")
    p.add_argument(
        "--axis_name",
        default="axial",
        type=str,
        # choices=axis_name_choices,
        help="Name of the axis to visualize. [%(default)s]",
    )
    p.add_argument(
        "--overlap_factor",
        nargs=2,
        metavar=("OVERLAP_HORIZ", "OVERLAP_VERT"),
        default=(0.6, 0.0),
        # type=ranged_type(float, 0.0, 1.0),
        help="The overlap factor with respect to the dimension. [%(default)s]",
    )
    p.add_argument(
        "--win_dims",
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(768, 768),
        type=int,
        help="The dimensions for the vtk window. [%(default)s]",
    )
    p.add_argument(
        "--vol_cmap_name",
        default=None,
        help="Colormap name for the volume image data. [%(default)s]",
    )
    p.add_argument(
        "--labelmap_cmap_name",
        default="viridis",
        help="Colormap name for the labelmap image data. [%(default)s]",
    )
    p.add_argument(
        "--atlas_name",
        # default="viridis",
        help="Atlas name to get the colormap for the label image. [%(default)s]",
    )

    return p


def _parse_args(parser):

    args = parser.parse_args()

    inputs = []
    output = []

    inputs.append(args.in_vol)
    inputs.append(args.in_transparency_mask)

    if args.in_labelmap:
        inputs.append(args.in_labelmap)

    output.append(args.out_fname)

    # assert_inputs_exist(parser, inputs)
    # assert_outputs_exist(parser, args, output)

    assert_same_resolution(inputs)

    return args


def _get_data_from_inputs(args):

    vol_img = nib.load(args.in_vol)
    mask_img = nib.load(args.in_transparency_mask)

    # Check header compatibility
    # if not is_header_compatible(vol_img, mask_img):
    #    raise ValueError(
    #        f"{args.in_vol} and {args.in_mask} do not have a compatible "
    #        f"header."
    #    )

    labelmap_img = None
    if args.in_labelmap:
        labelmap_img = nib.load(args.in_labelmap)

    #     # Check header compatibility
    #    if not is_header_compatible(vol_img, labelmap_img):
    #        raise ValueError(
    #            f"{args.in_vol} and {args.in_labelmap} do not have a "
    #            f"compatible header."
    #        )

    return vol_img, mask_img, labelmap_img


def main():

    parser = _build_arg_parser()
    args = _parse_args(parser)

    vol_img, mask_img, labelmap_img = _get_data_from_inputs(args)

    rows = args.mosaic_rows_cols[0]
    cols = args.mosaic_rows_cols[1]

    # Check if the mosaic can be built
    # check_slice_indices(vol_img, args.axis_name, args.slice_ids)
    # check_mosaic_layout(len(args.slice_ids), rows, cols)

    # Generate the images
    vol_scene_container = screenshot_slice(
        vol_img,
        Axis(args.axis_name),
        args.slice_ids,
        args.win_dims,
    )
    mask_scene_container = screenshot_slice(
        mask_img,
        Axis(args.axis_name),
        args.slice_ids,
        args.win_dims,
    )

    labelmap_scene_container = []
    labelmap_cmap_name = None
    if labelmap_img:
        # Set the interpolation to nearest when creating a labelmap scene, since
        # otherwise the default interpolation mode ("linear") creates
        # artefactual (interpolated) colors around the boundaries of each label.
        interpolation = "nearest"
        # ToDo
        # Workaround to make work cases where args.labelmap_cmap_name is a
        # mpl colormap name vs a LUT file.
        # Reconciling both ways requires solving the underlying issue about the
        # rgb2gray4pil call in compose_mosaic
        if Path(args.labelmap_cmap_name).exists():
            lookup_colormap = create_vtk_color_lut(args.labelmap_cmap_name)
        else:
            lookup_colormap = None
            labelmap_cmap_name = args.labelmap_cmap_name

        kwargs = dict(
            {
                "interpolation": interpolation,
                "lookup_colormap": lookup_colormap,
            }
        )
        labelmap_scene_container = screenshot_slice(
            labelmap_img,
            Axis(args.axis_name),
            args.slice_ids,
            args.win_dims,
            **kwargs,
        )

    # Compose the mosaic
    img = compose_mosaic(
        vol_scene_container,
        mask_scene_container,
        args.win_dims,
        rows,
        cols,
        args.overlap_factor,
        labelmap_scene_container=labelmap_scene_container,
        vol_cmap_name=args.vol_cmap_name,
        labelmap_cmap_name=labelmap_cmap_name,
    )

    # Save the mosaic
    img.save(args.out_fname)


if __name__ == "__main__":
    main()
