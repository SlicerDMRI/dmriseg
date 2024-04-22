# -*- coding: utf-8 -*-

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax

from dmriseg.image.utils import get_one_hot_2  # get_one_hot1, get_one_hot_k


class DiceLossK(nn.Module):
    """From
    https://github.com/kornia/kornia/blob/master/kornia/losses/dice.py#L77
    """

    def __init__(self):
        super(DiceLossK, self).__init__()

    def forward(self, y_pred, y_true):
        # Prepare inputs
        pred_soft: torch.Tensor = y_pred.softmax(dim=1)

        # ToDo
        # Keeps giving an  Expected dtype int64 for index at
        # _one_hot.scatter_(1, _labels.unsqueeze(1), 1.0)
        # create the labels one hot tensor
        # target_one_hot: torch.Tensor = get_one_hot_k(
        #    y_true, num_classes=y_pred.shape[1],  # device=y_pred.device,
        #    dtype=y_pred.dtype)

        target_one_hot = get_one_hot_2(y_true, y_pred.shape[1])

        # set dimensions for the appropriate averaging
        dims: tuple[int, ...] = (2, 3)
        # compute the actual dice score
        intersection = torch.sum(pred_soft * target_one_hot, dims)
        cardinality = torch.sum(pred_soft + target_one_hot, dims)

        dice_score = 2.0 * intersection / (cardinality + 1e-6)
        dice_loss = -dice_score + 1.0

        # reduce the loss across samples (and classes in case of `macro` averaging)
        dice_loss = torch.mean(dice_loss)

        return dice_loss


class DiceLoss(nn.Module):
    """
    From the paper "A Generalized Surface Loss for Reducing the Hausdorff
    Distance in Medical Imaging Segmentation": https://arxiv.org/pdf/2302.03868.pdf
    https://github.com/aecelaya/gen-surf-loss
    """

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-6
        self.axes = (1, 2, 3)  # self.axes = (2, 3, 4)

    def forward(self, y_true, y_pred):

        y_true = get_one_hot_2(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=0)

        num = torch.sum(torch.square(y_true - y_pred), dim=self.axes)
        den = (
            torch.sum(torch.square(y_true), dim=self.axes)
            + torch.sum(torch.square(y_pred), dim=self.axes)
            + self.smooth
        )

        loss = torch.mean(num / den, axis=0)  # , axis=1)
        loss = torch.mean(loss)
        return loss


class DiceCELoss(nn.Module):
    """
    From the paper "A Generalized Surface Loss for Reducing the Hausdorff
    Distance in Medical Imaging Segmentation": https://arxiv.org/pdf/2302.03868.pdf
    https://github.com/aecelaya/gen-surf-loss
    """

    def __init__(self):
        super(DiceCELoss, self).__init__()
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, y_true, y_pred):
        # Dice loss
        loss_dice = self.dice_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot_2(y_true, y_pred.shape[1]).to(torch.float32)

        # Cross entropy loss
        loss_ce = self.cross_entropy(y_pred, y_true)

        return loss_ce + loss_dice


class HDOneSidedLoss(nn.Module):
    """
    From the paper "A Generalized Surface Loss for Reducing the Hausdorff
    Distance in Medical Imaging Segmentation": https://arxiv.org/pdf/2302.03868.pdf
    https://github.com/aecelaya/gen-surf-loss
    """

    def __init__(self):
        super(HDOneSidedLoss, self).__init__()
        self.region_loss = DiceCELoss()
        self.alpha = 0.5

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot_2(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        # Compute boundary loss
        boundary_loss = torch.mean(
            torch.square(y_true - y_pred) * torch.square(dtm)
        )

        return alpha * region_loss + (1.0 - alpha) * boundary_loss


class GenSurfLoss(nn.Module):
    """
    From the paper "A Generalized Surface Loss for Reducing the Hausdorff
    Distance in Medical Imaging Segmentation": https://arxiv.org/pdf/2302.03868.pdf
    https://github.com/aecelaya/gen-surf-loss
    """

    def __init__(self, class_weights):
        super(GenSurfLoss, self).__init__()
        self.region_loss = DiceCELoss()

        # Define class weight scheme
        # Move weights to cuda if already given by user
        if class_weights is not None:
            self.class_weights = torch.Tensor(class_weights).to("cuda")
        else:
            self.class_weights = None

        self.smooth = 1e-6
        self.axes = (1, 2, 3)  # (2, 3, 4)

    def forward(self, y_true, y_pred, dtm, alpha):
        # Compute region based loss
        region_loss = self.region_loss(y_true, y_pred)

        # Prepare inputs
        y_true = get_one_hot_2(y_true, y_pred.shape[1])
        y_pred = softmax(y_pred, dim=1)

        if self.class_weights is None:
            class_weights = torch.sum(y_true, dim=self.axes)
            class_weights = 1.0 / (torch.square(class_weights) + 1.0)
        else:
            class_weights = self.class_weights

        # Compute boundary loss
        # Flip each one-hot encoded class
        y_worst = torch.square(1.0 - y_true)

        num = torch.sum(torch.square(dtm * (y_worst - y_pred)), axis=self.axes)
        num *= class_weights

        den = torch.sum(torch.square(dtm * (y_worst - y_true)), axis=self.axes)
        den *= class_weights
        den += self.smooth

        # boundary_loss = torch.sum(num, axis=1) / torch.sum(den, axis=1)
        boundary_loss = torch.sum(num, axis=0) / torch.sum(den, axis=0)
        boundary_loss = torch.mean(boundary_loss)
        boundary_loss = 1.0 - boundary_loss

        return alpha * region_loss + (1.0 - alpha) * boundary_loss


def soft_skeletonize(x, thresh_width=10):
    """Differentiable approximation of morphological skeletonization operation
    thresh_width - maximal expected width of vessel

    From the paper A Surprisingly Effective Perimeter-based Loss for Medical
    Image Segmentation: https://openreview.net/pdf?id=NDEmtyb4cXu
    """

    for i in range(thresh_width):
        min_pool_x = torch.nn.functional.max_pool2d(x * -1, (3, 3), 1, 1) * -1
        max_min_pool_x = torch.nn.functional.max_pool2d(
            min_pool_x, (3, 3), 1, 1
        )
        contour = torch.nn.functional.relu(max_min_pool_x - min_pool_x)
        x = torch.nn.functional.relu(x - contour)
    return x


class soft_cldice_loss:
    """inputs shape  (batch, channel, height, width).
    calculate clDice loss
    Because pred and target at moment of loss calculation will be a torch tensors
    it is preferable to calculate target_skeleton on the step of batch forming,
    when it will be in numpy array format by means of opencv

    From the paper A Surprisingly Effective Perimeter-based Loss for Medical
    Image Segmentation: https://openreview.net/pdf?id=NDEmtyb4cXu
    https://github.com/rosanajurdi/Prior-based-Losses-for-Medical-Image-Segmentation
    """

    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)
        b, _, w, h = pc.shape
        cl_pred = soft_skeletonize(pc)
        target_skeleton = soft_skeletonize(tc)
        big_pen: Tensor = (cl_pred - target_skeleton) ** 2
        contour_loss = big_pen / (w * h)

        return contour_loss.mean()
