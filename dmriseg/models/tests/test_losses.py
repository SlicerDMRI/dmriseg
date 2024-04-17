#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from monai.losses.dice import DiceLoss
from skimage.draw import disk

from dmriseg.image.utils import get_one_hot1
from dmriseg.models.losses import DiceCELoss as LocalDiceCELoss
from dmriseg.models.losses import DiceLoss as LocalDiceLoss
from dmriseg.models.losses import DiceLossK, GenSurfLoss, soft_cldice_loss


def test_soft_cldice_loss():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    y_pred_fg = torch.Tensor(mask)
    y_pred_bg = torch.logical_not(y_pred_fg)
    y_pred = torch.stack([y_pred_fg, y_pred_bg], dim=0).unsqueeze(0)
    # y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    kwargs = dict({"idc": [1]})
    contour = soft_cldice_loss(**kwargs)
    c_loss = contour(y_pred, y_pred)
    print(f"Contour loss: {c_loss}")  # 0 as expected

    _y_true = get_one_hot1(y_pred_fg.unsqueeze(0), 2)

    c_loss = contour(y_pred, _y_true)
    print(
        f"Contour loss: {c_loss}"
    )  # 4.234910011291504e-05 I could maybe understand

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    rr1, cc1 = disk((50, 50), 2)
    mask1[rr1, cc1] = 1
    y_pred_fg1 = torch.Tensor(mask1)
    y_pred_bg1 = torch.logical_not(y_pred_fg1)
    y_pred1 = torch.stack([y_pred_fg1, y_pred_bg1], dim=0).unsqueeze(0)

    c_loss = contour(y_pred1, _y_true)
    print(
        f"Contour loss: {c_loss}"
    )  # 5.628913640975952e-05 Does not make any sense

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    rr1, cc1 = disk((60, 60), 25)
    mask1[rr1, cc1] = 1
    y_pred_fg1 = torch.Tensor(mask1)
    y_pred_bg1 = torch.logical_not(y_pred_fg1)
    y_pred1 = torch.stack([y_pred_fg1, y_pred_bg1], dim=0).unsqueeze(0)

    c_loss = contour(y_pred1, _y_true)
    print(
        f"Contour loss: {c_loss}"
    )  # 4.234910011291504e-05 Does not make any sense


def test_dice_loss_monai():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    y_pred_fg = torch.Tensor(mask)
    # y_pred_bg = torch.logical_not(y_pred_fg)
    # y_pred = torch.stack([y_pred_fg, y_pred_bg], dim=0).unsqueeze(0)
    y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    dice = DiceLoss(reduction="none")
    dice_loss = dice(y_true, y_true)  # 0 as expected
    assert dice_loss.item() == 0
    print(f"Dice loss monai: {dice_loss.item()}")

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 96
    radius = 32
    rr, cc = disk((row, col), radius)
    mask1[rr, cc] = 1

    y_pred_fg1 = torch.Tensor(mask1)
    y_pred1 = y_pred_fg1.unsqueeze(0).unsqueeze(0)
    dice_loss = dice(y_pred1, y_true)
    print(f"Dice loss monai: {dice_loss.item()}")


def test_dice_loss_k():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    y_pred_fg = torch.Tensor(mask)
    y_pred_bg = torch.empty(y_pred_fg.size())
    # y_true_bg = torch.logical_not(y_true_fg)
    y_pred = torch.stack([y_pred_fg, y_pred_bg], dim=0).unsqueeze(0)
    y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    dice = DiceLossK()
    dice_loss = dice(y_pred.type(torch.float64), y_true.type(torch.float64))
    print(f"Dice loss kornia: {dice_loss.item()}")  # Changes across runs !!

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 96
    radius = 32
    rr, cc = disk((row, col), radius)
    mask1[rr, cc] = 1

    y_pred_fg1 = torch.Tensor(mask1)
    y_pred = torch.stack([y_pred_fg1, y_pred_bg], dim=0).unsqueeze(0)

    dice_loss = dice(y_pred.type(torch.float64), y_true.type(torch.float64))
    print(f"Dice loss kornia: {dice_loss.item()}")  # Changes across runs !!


def test_dice_loss_local():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    y_pred_fg = torch.Tensor(mask)
    y_pred_bg = torch.empty(y_pred_fg.size())
    # y_true_bg = torch.logical_not(y_true_fg)
    y_pred = torch.stack([y_pred_fg, y_pred_bg], dim=0).unsqueeze(0)
    y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    dice = LocalDiceLoss()
    dice_loss = dice(y_true, y_pred)
    print(f"Dice loss local: {dice_loss.item()}")  # 0.333 ????

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 96
    radius = 32
    rr, cc = disk((row, col), radius)
    mask1[rr, cc] = 1

    y_pred_fg1 = torch.Tensor(mask1)
    y_pred = torch.stack([y_pred_fg1, y_pred_bg], dim=0).unsqueeze(0)

    dice_loss = dice(y_true, y_pred)
    print(
        f"Dice loss local: {dice_loss.item()}"
    )  # 0.333 ???? Same as above ???


def test_dice_ce_loss_local():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    # y_true = torch.Tensor(mask).unsqueeze(0)
    # y_pred = torch.Tensor(mask).unsqueeze(0)
    y_pred_fg = torch.Tensor(mask)
    y_pred_bg = torch.empty(y_pred_fg.size())
    # y_true_bg = torch.logical_not(y_true_fg)
    y_pred = torch.stack([y_pred_fg, y_pred_bg], dim=0).unsqueeze(0)
    y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    dice_ce_loss = LocalDiceCELoss()
    loss_val = dice_ce_loss(y_true, y_pred)
    # The value will not be in the [0, 1] interval unless I weigh each component
    # (dice, ce) with 0.5
    # assert 0 <= loss_val <= 1
    print(f"DiceCE: {loss_val.item()}")


def test_dice_surf_loss():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    # y_true = torch.Tensor(mask).unsqueeze(0).unsqueeze(0)
    # y_pred = torch.Tensor(mask).unsqueeze(0).unsqueeze(0)
    y_pred_fg = torch.Tensor(mask)
    y_pred_bg = torch.empty(y_pred_fg.size())
    # y_true_bg = torch.logical_not(y_true_fg)
    y_pred = torch.stack([y_pred_fg, y_pred_bg], dim=0).unsqueeze(0)
    y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    dtm = torch.randn(1, 1, 128, 128)
    alpha = 0.1

    class_weights = None
    gen_surf_loss = GenSurfLoss(class_weights)
    loss_val = gen_surf_loss(y_true, y_pred, dtm, alpha)
    print(f"GenSurfLoss: {loss_val.item()}")
    # assert 0 <= loss_val <= 1


def test_hausdorff_loss():

    mask = np.zeros((128, 128), dtype=np.uint8)
    row = 64
    col = 64
    radius = 32
    rr, cc = disk((row, col), radius)
    mask[rr, cc] = 1

    y_pred_fg = torch.Tensor(mask)
    # y_pred_bg = torch.logical_not(y_pred_fg)
    y_true = y_pred_fg.unsqueeze(0).unsqueeze(0)

    from monai.losses.hausdorff_loss import HausdorffDTLoss

    haussforff_dt = HausdorffDTLoss()
    haussforff_dt_loss = haussforff_dt(y_true, y_true)  # 0 as expected
    print(f"haussforff_dt_loss: {haussforff_dt_loss}")

    mask1 = np.zeros((128, 128), dtype=np.uint8)
    rr1, cc1 = disk((row, col), 16)
    mask1[rr1, cc1] = 1
    y_pred_fg1 = torch.Tensor(mask1)
    y_pred1 = y_pred_fg1.unsqueeze(0).unsqueeze(0)
    haussforff_dt_loss1 = haussforff_dt(
        y_true, y_pred1
    )  # We see that this is not normalized (it is 26.64)
    print(f"haussforff_dt_loss1: {haussforff_dt_loss1}")

    mask2 = np.zeros((128, 128), dtype=np.uint8)
    rr2, cc2 = disk((row, col), 48)
    mask2[rr2, cc2] = 1
    y_pred_fg2 = torch.Tensor(mask2)
    y_pred2 = y_pred_fg2.unsqueeze(0).unsqueeze(0)
    haussforff_dt_loss2 = haussforff_dt(
        y_true, y_pred2
    )  # Looks to be between 0 and inf.
    print(f"haussforff_dt_loss2: {haussforff_dt_loss2}")
