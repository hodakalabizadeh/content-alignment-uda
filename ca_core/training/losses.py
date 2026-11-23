"""Loss functions for Content Alignment training."""
import torch
import torch.nn as nn


class SmoothnessLoss:
    """L2 smoothness penalty on a deformation field: mean squared difference between neighbouring voxels.

    Works on 2D (B, C, H, W) and 3D (B, C, H, W, D) fields. The per-axis means are summed and divided
    by 2 in both cases (VoxelMorph divides by the number of axes); kept as trained, so in 3D the
    smoothness weight is effectively 1.5x the VoxelMorph convention.
    """

    def __call__(self, field):
        # |diff| * |diff| rather than diff ** 2: same value, but kept for bit-identical fp16 gradients
        dy = torch.abs(field[:, :, 1:, :] - field[:, :, :-1, :])
        dx = torch.abs(field[:, :, :, 1:] - field[:, :, :, :-1])
        d = torch.mean(dx * dx) + torch.mean(dy * dy)
        if field.dim() == 5:
            dz = torch.abs(field[:, :, :, :, 1:] - field[:, :, :, :, :-1])
            d = d + torch.mean(dz * dz)
        return d / 2.0


class DiceLoss(nn.Module):
    """Soft Dice loss, 1 - mean over classes of the Dice computed over the whole batch.

    pred: probabilities (B, n_classes, *spatial), e.g. the segmenter's softmax output.
    target: one-hot label of the same shape.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        target = target.type(pred.type())
        dims = (0,) + tuple(range(2, target.ndimension()))  # sum over batch and space, per class
        intersection = torch.sum(pred * target, dims)
        cardinality = torch.sum(pred + target, dims)
        dice = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice
